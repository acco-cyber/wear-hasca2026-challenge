"""Within-subject separability of variants: K-means / GMM / spectral clustering of TRUE family windows, cluster purity
vs. the e7 labels' within-family accuracy on the same windows."""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive")
from tlib import build_X, pca_reduce, std_within
from sklearn.cluster import KMeans, SpectralClustering
from scipy.optimize import linear_sum_assignment
C = pickle.load(open(r"E:\Claude code\wear\exp\order\cache_e7.pkl", "rb"))
F = {}
for w in ("eval", "extra", "extra2"):
    for s, d in pickle.load(open(rf"E:\Claude code\wear\exp\transductive\cache_{w}.pkl", "rb")).items(): F[s] = d["F"]
FAMS = {"jog": [1, 2, 3, 4, 5], "str": [6, 7, 8, 9, 10], "push": [11, 12], "sit": [13, 14], "lun": [16, 17]}
def matched_acc(yt, cl, K, cls):
    Cm = np.zeros((K, len(cls)))
    for i in range(K):
        for j, c in enumerate(cls): Cm[i, j] = np.sum((cl == i) & (yt == c))
    r, c = linear_sum_assignment(-Cm); return Cm[r, c].sum() / len(yt)
agg = {}
for s, d in C.items():
    y = d["y"]; lab = d["lab1"]
    feats = {"X128": d["X"]}
    Xv = std_within(np.asarray(F[s]["v768"], np.float64))
    for f, cls in FAMS.items():
        idx = np.where(np.isin(y, cls))[0]; yt = y[idx]; K = len(cls)
        e7acc = np.mean(lab[idx] == yt)
        agg.setdefault((f, "e7"), []).append(e7acc)
        # family-local PCA of v768
        Z = pca_reduce(Xv[idx], 16); Z2 = pca_reduce(Xv[idx], 4)
        for name, Xf in (("X128", d["X"][idx]), ("fpca16", Z), ("fpca4", Z2)):
            km = KMeans(K, n_init=5, random_state=0).fit(Xf); agg.setdefault((f, "km_" + name), []).append(matched_acc(yt, km.labels_, K, cls))
for f in FAMS:
    print(f, " ".join(f"{k[1]} {np.mean(v):.3f}" for k, v in agg.items() if k[0] == f))
