"""TEST structure with the purity cut baked in (work/test_structure.pkl format; sc of cut edges = -50 so cut(thr=-6) drops them).
Purity is computed from the lgbm_v1/fusion_v1 0.8/0.2 test blend (= the distribution the purity model was trained on), so the
structure is fixed and any probability file can be decoded with it (e.g. exp/decoder/decoder.py --struct <this> --probs X.npy --variant mrf4).
python make_test_cut.py 0.5,0.6"""
import sys as _s
from lk import *
from purity2 import edge_feats2

taus = [float(x) for x in (_s.argv[1] if len(_s.argv) > 1 else "0.5,0.6").split(",")]
T = pickle.load(open(os.path.join(WORK, "test_structure.pkl"), "rb")); model = pickle.load(open(os.path.join(EXP, "purity2_model.pkl"), "rb"))
P = blend_test(0.2); outs = {t: {} for t in taus}
for s, st in T.items():
    idx = st["idx"]; a, b, succ, F, Pg = edge_feats2(P[idx], dict(st, Lm=st["Lm"].astype(np.float32))); p = model.predict_proba(F)[:, 1]
    for t in taus:
        sc2 = np.array(st["sc"], np.float32).copy(); sc2[a[p < t]] = -50.0
        outs[t][s] = dict(idx=idx, cand=st["cand"], lo=st["lo"], succ0=st["succ0"], sc=sc2, Lm=st["Lm"])
        print(f"sbj {s} tau {t}: edges {len(a)} cut {int((p < t).sum())}", flush=True)
for t in taus:
    fn = os.path.join(EXP, f"test_structure_links_cut{t}.pkl"); pickle.dump(outs[t], open(fn, "wb")); print("wrote", fn)
