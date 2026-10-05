"""Cache the late-fusion inputs of K7+K9 (same arithmetic as v4_combine.py, equal weights) for fast count experiments."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
from v4_local import H, macro_f1

K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.npz")

st = [np.load(os.path.join(d, "stage.npz"), allow_pickle=True) for d in (K7, K9)]
y = st[0]["oof_y"].astype(np.int64); sbj = st[0]["oof_sbj"].astype(np.int64); fold = st[0]["oof_fold"].astype(np.int64)
tsbj = st[0]["test_sbj"].astype(np.int64)
assert (st[1]["oof_y"] == y).all() and (st[1]["test_sbj"] == tsbj).all()
Pos = [s["PB_OOF"].astype(np.float64) for s in st]; Pts = [s["PB_TEST"].astype(np.float64) for s in st]
geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(sum(0.5 * np.log(np.clip(M, 1e-9, None)) for M in Ms)))
Po, Pt = geo(Pos), geo(Pts)
Bo = H.lsm(sum(0.5 * s["B2_OOF"].astype(np.float64) for s in st)); Bt = H.lsm(sum(0.5 * s["B2_TEST"].astype(np.float64) for s in st))
np.savez_compressed(OUT, y=y, sbj=sbj, fold=fold, tsbj=tsbj, Po=Po, Pt=Pt, Bpo=np.exp(Bo), Bpt=np.exp(Bt),
                    P7o=Pos[0], P7t=Pts[0], P9o=Pos[1], P9t=Pts[1],
                    B7o=np.exp(st[0]["B2_OOF"].astype(np.float64)), B7t=np.exp(st[0]["B2_TEST"].astype(np.float64)),
                    B9o=np.exp(st[1]["B2_OOF"].astype(np.float64)), B9t=np.exp(st[1]["B2_TEST"].astype(np.float64)),
                    Q7o=st[0]["QB_OOF"].astype(np.float64), Q9o=st[1]["QB_OOF"].astype(np.float64),
                    Q7t=st[0]["QB_TEST"].astype(np.float64), Q9t=st[1]["QB_TEST"].astype(np.float64))
print("n oof", len(y), "n test", len(tsbj), "subjects", np.unique(sbj).tolist(), "test", np.unique(tsbj).tolist())
print("folds", {int(f): np.unique(sbj[fold == f]).tolist() for f in range(5)})
print("fused P argmax F1", round(macro_f1(y, Po.argmax(1)), 4))
