"""TEST: old link structure (work/test_structure.pkl) + purity-cut chains + lead's new decoder -> submission.
python predict_links.py --variant purity [--p1 work/lgbm_v1/test.npy --p2 work/fusion_v1/test.npy --w2 0.2] [--tau 0.5]
  (or --probs some_blend.npy : an already-blended (12234,19) probability array; overrides --p1/--p2)
Writes subs/sub_links_<variant>.csv, exp/links/test_probs_<variant>.npy, exp/links/test_structure_links.pkl (struct + cut succ).
"""
import argparse
from lk import *
from purity import decode_pure

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True)
    ap.add_argument("--p1", default=os.path.join(WORK, "lgbm_v1", "test.npy")); ap.add_argument("--p2", default=os.path.join(WORK, "fusion_v1", "test.npy"))
    ap.add_argument("--w2", type=float, default=0.2); ap.add_argument("--probs", default=None); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--model", default=None); ap.add_argument("--struct", default=os.path.join(WORK, "test_structure.pkl"))
    ap.add_argument("--v2", default=None, help="purity2 mode: 'hard' or 'soft_1.0' (uses purity2_model.pkl + chain-context features)")
    a = ap.parse_args()
    if a.v2:
        from purity2 import decode2
        a.model = a.model or os.path.join(EXP, "purity2_model.pkl")
        def decode_pure(P_, st_, m_, tau_, return_parts=True): return decode2(P_, st_, m_, tau_, a.v2, return_parts=True)
    else:
        a.model = a.model or os.path.join(EXP, "purity_model.pkl")
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    if a.probs: P = np.load(a.probs).astype(np.float64); P = P / P.sum(1, keepdims=True)
    else: P = blend_test(a.w2, a.p1, a.p2)
    np.save(os.path.join(EXP, f"test_probs_{a.variant}.npy"), P.astype(np.float32))
    T = pickle.load(open(a.struct, "rb")); model = pickle.load(open(a.model, "rb"))
    raw = P.argmax(1); lab = raw.copy(); out = {}
    for s, st in T.items():
        st = dict(st, Lm=st["Lm"].astype(np.float32))
        idx = st["idx"]; l, succ2, p = decode_pure(P[idx], st, model, a.tau, return_parts=True); lab[idx] = l
        out[s] = dict(st, Lm=st["Lm"].astype(np.float16), succ_cut=succ2, p_same=p)
        print(f"sbj {s}: n={len(idx)} edges kept {int((succ2>=0).sum())} | null raw {np.mean(raw[idx]==0):.3f} -> {np.mean(l==0):.3f} | counts {np.bincount(l, minlength=19).tolist()}", flush=True)
    pickle.dump(out, open(os.path.join(EXP, f"test_structure_links_{a.variant}.pkl"), "wb"))
    np.save(os.path.join(EXP, f"test_labels_{a.variant}.npy"), lab)
    outp = os.path.join(W, "subs", f"sub_links_{a.variant}.csv")
    df = pd.DataFrame({"id": tm.id, "target_feature": lab}); assert len(df) == 12234 and (np.sort(df.id.values) == np.arange(12234)).all()
    df.to_csv(outp, index=False); print("wrote", outp, "null", round(float((lab == 0).mean()), 3))
