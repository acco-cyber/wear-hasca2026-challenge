"""Bout-group reconstruction quality: components of (chain edges + mutual kNN edges) within each decoded family."""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\src")
from tlib import load_structs, build_X, session_P
from refine_core import knn_graph
from chain import cut
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
C, S = {}, {}
for w in ("eval", "extra", "extra2"):
    C.update(pickle.load(open(rf"E:\Claude code\wear\exp\transductive\cache_{w}.pkl", "rb"))); S.update(load_structs(w))
FAMS = {"jog": [1, 2, 3, 4, 5], "str": [6, 7, 8, 9, 10], "push": [11, 12], "sit": [13, 14], "lun": [16, 17]}
from sklearn.neighbors import NearestNeighbors
for (tc, kk, tk) in [(-2.0, 5, 0.0), (-2.0, 5, 0.5), (0.0, 3, 0.5), (-4.0, 10, 0.3)]:
    agg = {f: [0, 0, 0] for f in FAMS}
    for s in R:
        st = S[s]; d = C[s]; y = R[s]["y"]; lab = R[s]["lab"]; n = len(y)
        X = build_X(d["F"], "v768", 128, 32); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        succ = st["succ0"]; sc = st["sc"]
        for f, cls in FAMS.items():
            idx = np.where(np.isin(lab, cls))[0]
            if len(idx) < 5: continue
            loc = -np.ones(n, int); loc[idx] = np.arange(len(idx))
            r_, c_ = [], []
            a = idx[(sc[idx] >= tc)]; b = succ[a]; m = loc[b] >= 0; r_ += list(loc[a[m]]); c_ += list(loc[b[m]])
            nn = NearestNeighbors(n_neighbors=min(kk + 1, len(idx))).fit(Xn[idx]); dist, nb = nn.kneighbors(Xn[idx])
            sim = 1 - dist[:, 1:] ** 2 / 2; nb = nb[:, 1:]
            A = np.zeros((len(idx), len(idx)), bool)
            for i in range(len(idx)):
                for j, sv in zip(nb[i], sim[i]):
                    if sv >= tk: A[i, j] = True
            M = A & A.T; ii, jj = np.where(M); r_ += list(ii); c_ += list(jj)
            G = csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(idx), len(idx)))
            ncomp, comp = connected_components(G, directed=False)
            yt = y[idx]; pur = 0
            for k in range(ncomp):
                mm = comp == k; pur += np.bincount(yt[mm], minlength=19).max()
            agg[f][0] += pur; agg[f][1] += len(idx); agg[f][2] += ncomp
    print(f"chain thr {tc} knn k {kk} sim>= {tk}: " + " | ".join(f"{f} purity {v[0]/max(v[1],1):.3f} comps/sbj {v[2]/18:.1f}" for f, v in agg.items()), flush=True)
