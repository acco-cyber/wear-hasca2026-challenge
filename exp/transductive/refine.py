"""Test CLI: per-subject transductive kNN refinement on top of a decoder.

Decoder "lead" (default): lead's graph-smooth + chain-Viterbi + count-calibrated decoder (null x0.5, counts 80..250, p_stay 0.8).
Decoder "mrf4": exp/decoder/decoder.py decode_subject(P, st, VARIANTS['mrf4']) (imported read-only).

  python refine.py --tags lgbm_v1,fusion_v1 --weights 0.8,0.2 --variant knn_w6                 (lead decoder)
  python refine.py --probs <test_probs.npy (12234,19)> --decoder mrf4 --variant <name> [--steps '[{"w":6,"d_vid":128}]']
  add --baseline_only to write the un-refined decode.
Inputs: work/test_structure.pkl, data/prep/test_vid_pca.npy, data/prep/test_vid_mean768.npy, data/test/test_inertial_data.npy,
        data/test/test_meta_data.csv
Writes: subs/sub_transductive_<variant>.csv (lead decoder) or subs/sub_transductive_mrf4_<variant>.csv (mrf4), and
        exp/transductive/test_probs_<...>.npy = the refined per-window probabilities fed to the final decode.
"""
import os, sys, json, argparse, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tlib import *
from refine_core import refine, knn_label_Q, DEFAULT

SUBS = os.path.join(ROOT, "subs"); LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
# best sim config with the lead decoder (REPORT.md): spread the decoded labels of ALL the subject's windows over a
# within-subject video kNN graph (k=5, cosine on subject-standardised mean-768 VideoMAE -> PCA-128 whitened), label
# spreading alpha 0.9, Q smoothed with eps 0.1, logP + 6*logQ, re-decode. sim: eval +0.041, extra +0.040, extra2 +0.029.
BEST = dict(clf="knn", spec="v768", d_vid=128, k=5, alpha=0.9, sel="all", w=6.0, eps=0.1, rounds=1)
# best with mrf4 (bl_v3b_v1_f OOF): one step, same kNN, w=6, PCA-128. sim: eval +0.011, dec-extra +0.008, 18/18 up.
BEST_MRF4 = [dict(w=6.0, d_vid=128, k=5, alpha=0.9, eps=0.1)]

def load_probs(tags, weights):
    L = sum(w * np.log(np.clip(np.load(os.path.join(WORK, t, "test.npy")).astype(np.float64), 1e-6, 1)) for t, w in zip(tags, weights))
    P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)

def mrf4_refine(P, F, st, steps, baseline_only=False):
    sys.path.insert(0, os.path.join(ROOT, "exp", "decoder"))
    from decoder import decode_subject, VARIANTS
    cfg = VARIANTS[os.environ.get("DVARIANT", "mrf4")]; lab0 = decode_subject(P, st, cfg, None); lab = lab0; Pn = P
    if baseline_only: return lab0, lab0, P
    L = np.log(np.clip(P, 1e-6, 1)); Xc = {}
    for s in steps:
        d = s.get("d_vid", 64)
        if d not in Xc: Xc[d] = build_X(F, "v768", d, 32)
        Q = knn_label_Q(Xc[d], lab, s.get("k", 5), s.get("alpha", 0.9), s.get("eps", 0.1))
        L = L + s["w"] * np.log(Q); Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        lab = decode_subject(Pn, st, cfg, None)
    return lab, lab0, Pn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=None); ap.add_argument("--tags", default="lgbm_v1,fusion_v1"); ap.add_argument("--weights", default="0.8,0.2")
    ap.add_argument("--variant", required=True); ap.add_argument("--cfg", default=None, help="JSON overrides of the refinement config (lead decoder)")
    ap.add_argument("--decoder", default="lead", choices=["lead", "mrf4"]); ap.add_argument("--steps", default=None, help="JSON list of kNN steps (mrf4)")
    ap.add_argument("--baseline_only", action="store_true")
    a = ap.parse_args()
    if a.probs: P_all = np.load(a.probs).astype(np.float64); P_all /= P_all.sum(1, keepdims=True)
    else: P_all = load_probs(a.tags.split(","), [float(w) for w in a.weights.split(",")])
    assert P_all.shape == (12234, NC), P_all.shape
    cfg = {**DEFAULT, **BEST, **(json.loads(a.cfg) if a.cfg else {})}; steps = json.loads(a.steps) if a.steps else BEST_MRF4
    print("decoder", a.decoder, "config:", json.dumps(steps if a.decoder == "mrf4" else cfg), flush=True)
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv")); assert (tm.id.to_numpy() == np.arange(len(tm))).all()
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)   # train quantisation
    vp = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r"); v768 = np.load(os.path.join(PREP, "test_vid_mean768.npy"), mmap_mode="r")
    limb = np.array([LIMBS.index(l) for l in tm.sensor_location])
    struct = pickle.load(open(os.path.join(WORK, "test_structure.pkl"), "rb"))
    lab = P_all.argmax(1).copy(); Pout = P_all.copy()
    for s, st in struct.items():
        t0 = time.time(); idx = st["idx"]; n = len(idx); P = P_all[idx]
        F = window_features(vp[idx], v768[idx], xi[idx], limb[idx])
        if a.decoder == "mrf4":
            l, lab0, Pn = mrf4_refine(P, F, st, steps, a.baseline_only)
        else:
            g, chains = prep_decoder(st, n); lab0, Pg0, _ = decode(P, g, chains)
            if a.baseline_only: l = lab0; Pn = P
            else: labs, Pn = refine(P, F, g, chains, lab0, Pg0, cfg); l = labs[-1]
        lab[idx] = l; Pout[idx] = Pn
        print(f"sbj {s}: n={n} changed vs un-refined {np.mean(l != lab0):.3f} | null {np.mean(lab0==0):.3f}->{np.mean(l==0):.3f} | "
              f"counts {np.bincount(l, minlength=NC).tolist()} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(SUBS, exist_ok=True); pre = "mrf4_" if a.decoder == "mrf4" else ""
    out = os.path.join(SUBS, f"sub_transductive_{pre}{a.variant}.csv")
    pd.DataFrame({"id": tm.id, "target_feature": lab}).to_csv(out, index=False)
    np.save(os.path.join(TD, f"test_probs_{pre}{a.variant}.npy"), Pout.astype(np.float32))
    print("wrote", out, "| overall null", round(float((lab == 0).mean()), 3))

if __name__ == "__main__":
    main()
