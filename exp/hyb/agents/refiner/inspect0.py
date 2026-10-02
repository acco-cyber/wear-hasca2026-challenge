import os, sys, pickle
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP
sm = {k: v for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
for k, v in sm.items():
    print(k, v.shape, v.dtype, v[:5])
print("sbj unique", np.unique(sm["sbj"]).tolist())
print("rec unique", len(np.unique(sm["rec"])))
for f in range(5):
    m = sm["fold"] == f
    print("fold", f, "sbj", np.unique(sm["sbj"][m]).tolist(), "n", int(m.sum()))
rows = np.load(r"E:\Claude code\wear\exp\hyb\rows.npz"); print(list(rows.keys()))
o2t = rows["ours_to_theirs"]
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
has = np.zeros(len(sm["y"]), bool)
for s, d in R.items():
    ii = o2t[int(d["a"]):int(d["a"]) + int(d["n"])]; has[ii] = True
    print("session", s, "n", d["n"], "sbj", np.unique(sm["sbj"][ii]).tolist(), "rec", np.unique(sm["rec"][ii]).tolist(), "fold", np.unique(sm["fold"][ii]).tolist())
print("has", has.mean(), has.sum())
bl = np.load(os.path.join(KEEP, "blend.npz")); print({k: bl[k].shape for k in bl.keys()})
