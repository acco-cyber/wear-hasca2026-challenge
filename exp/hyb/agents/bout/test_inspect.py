import os
import numpy as np, pandas as pd
from common import *

bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64)
Pt = np.load(r"E:\Claude code\wear\subs\sub_gl6_p03c03_top2_055_P.npy").astype(np.float64); Pt /= Pt.sum(1, keepdims=True)
sub = pd.read_csv(r"E:\Claude code\wear\subs\sub_gl6_p03c03_top2_055.csv").sort_values("id").target_feature.to_numpy()
ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True)
print("ids == arange", (ids == np.arange(len(ids))).all(), "n", len(sub))
f0 = finish(Pt, dict(sbj=sbj, sets={}))
print("agreement finish(P,default targets) vs submitted", np.mean(f0 == sub))
for s in np.unique(sbj):
    m = sbj == s
    print("sbj", s, "n", m.sum(), "null share sub", round(np.mean(sub[m] == 0), 3), "finish", round(np.mean(f0[m] == 0), 3), "agree", round(np.mean(f0[m] == sub[m]), 3))
l0 = np.load(os.path.join(KEEP, "links_L0.npz")); l2 = np.load(os.path.join(KEEP, "links_L2_test.npz"))
for nm, su, sc in (("L0", l0["test_succ"], l0["test_score"]), ("L2", l2["succ"], l2["score_qn"])):
    m = su >= 0
    print(nm, "has succ", m.mean(), "same sub-label", np.mean(sub[m] == sub[su[m]]), "score q", np.quantile(sc[m], [.1, .5, .9]))
m = (l0["test_succ"] >= 0) & (l2["succ"] >= 0); print("L0/L2 same succ", np.mean(l0["test_succ"][m] == l2["succ"][m]))
# OOF analogue: same-pred-label rate for L0 links on OOF
d = load_oof(); pred = finish(d["P"], dict(sbj=d["sbj"], sets=TRAIN_SETS)); m = d["succ"] >= 0
print("OOF L0 same-pred-label", np.mean(pred[m] == pred[d["succ"][m]]), "same-true-label", np.mean(d["y"][m] == d["y"][d["succ"][m]]))
