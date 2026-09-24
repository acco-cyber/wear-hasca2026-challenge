"""Test-side: build the links structure for the 4 test subjects (encoder + v2 scorer) and decode a probability file.
python build_test.py --enc enc_A.pt[,enc_B.pt] --scorer scorer_v2.pkl --struct test_structure_links.pkl --variant v2
                     [--p1 work/lgbm_v1/test.npy --p2 work/fusion_v1/test.npy --w2 0.2] [--thr -6 --k 10 --alpha 0.5 --iters 5 --ps 0.8]
The structure pickle is cached: rerunning with other --p1/--p2 only redoes the decoding.
"""
import argparse, time
from lk import *
from feat2 import pair_features2
from run2 import load_models, enc_scores, logodds

def build_struct(enc_names, scorer_path, out_path):
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)   # match train quantisation
    vid = np.load(os.path.join(PREP, "test_vid_pca.npy"))
    limb = np.array([LIMBS.index(l) for l in tm.sensor_location])
    models = load_models(enc_names); scorer = pickle.load(open(scorer_path, "rb")); out = {}
    for s in sorted(tm.sbj_id.unique()):
        idx = np.where(tm.sbj_id.to_numpy() == s)[0]; n = len(idx); t0 = time.time()
        V = np.asarray(vid[idx], np.float32); Snn = enc_scores(models, V, xi[idx], limb[idx])
        cand, F = pair_features2(V, xi[idx], limb[idx], Snn); del Snn
        lo = logodds(scorer, F); del F
        succ0, scs, Lm = assignment_from(cand, lo, n)
        out[s] = dict(idx=idx, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm.astype(np.float16))
        print(f"sbj {s}: n={n} M={cand.shape[1]} ({time.time()-t0:.0f}s); median top logodds {np.median(lo.max(1)):.2f}", flush=True)
    pickle.dump(out, open(out_path, "wb")); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True); ap.add_argument("--scorer", required=True); ap.add_argument("--struct", default="test_structure_links.pkl")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--p1", default=os.path.join(WORK, "lgbm_v1", "test.npy")); ap.add_argument("--p2", default=os.path.join(WORK, "fusion_v1", "test.npy")); ap.add_argument("--w2", type=float, default=0.2)
    ap.add_argument("--thr", type=float, default=-6.0); ap.add_argument("--k", type=int, default=10); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5); ap.add_argument("--ps", type=float, default=0.8); ap.add_argument("--lo", type=int, default=60); ap.add_argument("--hi", type=int, default=160)
    a = ap.parse_args()
    sp = a.struct if os.path.isabs(a.struct) else os.path.join(EXP, a.struct)
    scp = a.scorer if os.path.isabs(a.scorer) else os.path.join(EXP, a.scorer)
    st_all = pickle.load(open(sp, "rb")) if os.path.exists(sp) else build_struct(a.enc.split(","), scp, sp)
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    P = blend_test(a.w2, a.p1, a.p2); np.save(os.path.join(EXP, f"test_probs_{a.variant}.npy"), P.astype(np.float32))
    raw = P.argmax(1); lab = raw.copy()
    for s, st in st_all.items():
        idx = st["idx"]
        l = decode(P[idx], st["cand"], st["lo"], st["succ0"], st["sc"], st["Lm"].astype(np.float32), thr=a.thr, k=a.k, alpha=a.alpha, iters=a.iters,
                   lo_c=a.lo, hi_c=a.hi, p_stay=a.ps)
        lab[idx] = l
        print(f"sbj {s}: n={len(idx)} null raw {np.mean(raw[idx]==0):.3f} -> {np.mean(l==0):.3f} changed {np.mean(l!=raw[idx]):.3f} counts {np.bincount(l, minlength=19).tolist()}", flush=True)
    np.save(os.path.join(EXP, f"test_labels_{a.variant}.npy"), lab)
    outp = os.path.join(W, "subs", f"sub_links_{a.variant}.csv")
    pd.DataFrame({"id": tm.id, "target_feature": lab}).to_csv(outp, index=False); print("wrote", outp, "rows", len(lab), "null", round(float((lab == 0).mean()), 3))
