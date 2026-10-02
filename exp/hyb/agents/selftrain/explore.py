import os, sys, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP, HB, N_CLS
sm = {k: v for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
for k, v in sm.items():
    print(k, v.shape, v.dtype, v[:5])
y, sbj, fold = sm["y"].astype(int), sm["sbj"].astype(int), sm["fold"].astype(int)
for f in range(5):
    m = fold == f
    print("fold", f, "n", m.sum(), "subjects", np.unique(sbj[m]).tolist())
l0 = np.load(os.path.join(KEEP, "links_L0.npz")); print({k: l0[k].shape for k in l0.files})
succ = l0["oof_succ"].astype(np.int64)
print("L0 succ coverage", np.mean(succ >= 0), "same-label", np.mean(y[succ[succ >= 0]] == y[succ >= 0]))
# chain lengths
n = len(succ); prv = np.full(n, -1); prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
heads = np.flatnonzero(prv < 0); lens = []
for h in heads:
    L = 1; c = h
    while succ[c] >= 0 and L < 100000:
        c = succ[c]; L += 1
    lens.append(L)
lens = np.array(lens); print("chains", len(lens), "len quantiles", np.percentile(lens, [0, 25, 50, 75, 90, 99, 100]), "cover", lens.sum())
print("in-degree>1:", np.sum(np.bincount(succ[succ >= 0], minlength=n) > 1))
# test
bl = np.load(os.path.join(KEEP, "blend.npz")); print(bl.files)
ts = bl["test_sbj"].astype(int); print("test subjects", np.unique(ts, return_counts=True))
l2 = np.load(os.path.join(KEEP, "links_L2_test.npz")); print({k: l2[k].shape for k in l2.files})
rows = np.load(os.path.join(r"E:\Claude code\wear\exp\hyb", "rows.npz")); print({k: rows[k].shape for k in rows.files})
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb")); print(len(R), list(R.items())[0][0], list(list(R.items())[0][1].keys()))
o2t = rows["ours_to_theirs"]; ours = np.full(len(y), -1)
for s, d in R.items():
    ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
has = ours >= 0
print("has frac", has.mean(), "acc ours on has", np.mean(ours[has] == y[has]))
for s in np.unique(sbj):
    m = sbj == s
    print("sbj", s, "fold", np.unique(fold[m]), "n", m.sum(), "has", round(has[m].mean(), 3), "null frac", round(np.mean(y[m] == 0), 3))
fo = np.load(os.path.join(KEEP, "feat_oof.npz")); print({k: fo[k].shape for k in fo.files})
