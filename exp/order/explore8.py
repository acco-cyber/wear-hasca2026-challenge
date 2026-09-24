"""Within-family edge purity: link-scorer edges (succ0 / top-k cand) and video kNN edges between windows of the same
family: fraction with same true label (variant) and same true set."""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\src")
from tlib import load_structs, build_X
from sklearn.neighbors import NearestNeighbors
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
C, S = {}, {}
for w in ("eval", "extra", "extra2"):
    C.update(pickle.load(open(rf"E:\Claude code\wear\exp\transductive\cache_{w}.pkl", "rb"))); S.update(load_structs(w))
FAM = np.zeros(19, int); FAM[1:6] = 1; FAM[6:11] = 2; FAM[11:13] = 3; FAM[13:15] = 4; FAM[15] = 5; FAM[16:18] = 6; FAM[18] = 7
MULTI = [1, 2, 3, 4, 6]
stats = {}
def add(k, same_lab, same_set):
    stats.setdefault(k, [0, 0, 0]); stats[k][0] += same_lab.sum(); stats[k][1] += same_set.sum(); stats[k][2] += len(same_lab)
for s in R:
    st = S[s]; d = C[s]; y = R[s]["y"]; n = len(y)
    setid = np.r_[0, np.cumsum(np.diff(y) != 0)]
    fy = FAM[y]
    def ev(a, b, k):
        m = np.isin(fy[a], MULTI) & (fy[a] == fy[b])
        add(k, (y[a] == y[b])[m], (setid[a] == setid[b])[m])
    a = np.arange(n); b = st["succ0"]; sc = st["sc"]
    for t in (-6, -2, 0, 2):
        m = sc >= t; ev(a[m], b[m], f"succ0 sc>={t}")
    cand = st["cand"]; lo = st["lo"]
    for kk in (1, 3, 10):
        o = np.argsort(-lo, 1)[:, :kk]; bb = cand[np.arange(n)[:, None], o]; aa = np.repeat(np.arange(n)[:, None], kk, 1)
        m = bb >= 0; ev(aa[m], bb[m], f"cand top{kk}")
    X = build_X(d["F"], "v768", 128, 32); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    nn = NearestNeighbors(n_neighbors=11).fit(Xn); _, nb = nn.kneighbors(Xn)
    for kk in (1, 5, 10):
        bb = nb[:, 1:kk + 1]; aa = np.repeat(np.arange(n)[:, None], kk, 1); ev(aa.ravel(), bb.ravel(), f"vknn k{kk}")
    # IMU-only kNN (same limb only)
    Fi = np.asarray(d["F"]["imu"], np.float64); lb = np.asarray(d["F"]["limb"])
for k, v in stats.items(): print(f"{k:16s} same-variant {v[0]/v[2]:.3f} same-set {v[1]/v[2]:.3f} (n={v[2]})")
