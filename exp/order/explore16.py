"""Balanced spectral bisection of a 2-variant family's windows on the link (+kNN) graph: can it separate the variants
better than e7 does? Windows = true family AND e7 family (comparable accuracies)."""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\exp\decoder")
from tlib import load_structs
from common import graph_matrix
from refine_core import knn_graph
C = pickle.load(open(r"E:\Claude code\wear\exp\order\cache_e7.pkl", "rb"))
S = {}
for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
FAMS = {"push": [11, 12], "sit": [13, 14], "lun": [16, 17]}
agg = {}
for s, d in C.items():
    st = S[s]; y = d["y"]; lab = d["lab1"]; n = len(y)
    Wl = graph_matrix(st["cand"], st["lo"], n, k=10); Wl = (Wl + Wl.T).tocsr()
    _, Wk = knn_graph(d["X"], 5); Wk = Wk.tocsr()
    line = []
    for f, cls in FAMS.items():
        idx = np.where(np.isin(y, cls) & np.isin(lab, cls))[0]
        if len(idx) < 20: continue
        yt = y[idx]; e7acc = np.mean(lab[idx] == yt)
        res = [f"{f}: e7 {e7acc:.2f}"]
        for name, W in (("link", Wl), ("knn", Wk), ("both", Wl + Wk)):
            Ws = W[idx][:, idx]; dg = np.asarray(Ws.sum(1)).ravel() + 1e-9; Dm = sp.diags(1 / np.sqrt(dg))
            vals, vecs = np.linalg.eigh((Dm @ Ws @ Dm).toarray()); o = np.argsort(-vals); v = vecs[:, o[1]] / np.sqrt(dg)
            part = v > np.median(v)
            acc = max(np.mean((part == (yt == cls[1]))), np.mean((part == (yt == cls[0]))))
            # assignment by classifier: which part gets which variant
            L = d["L1"][idx]; sA = L[part][:, cls].sum(0); sB = L[~part][:, cls].sum(0)
            asg = (sA[1] - sA[0]) > (sB[1] - sB[0])     # part A -> cls[1]
            pred = np.where(part == asg, cls[1], cls[0]); acc_asg = np.mean(pred == yt)
            res.append(f"{name} {acc:.2f}/{acc_asg:.2f}")
            agg.setdefault((f, name), []).append((e7acc, acc, acc_asg))
        line.append(" ".join(res))
    print(s, " | ".join(line), flush=True)
for k, v in agg.items(): v = np.array(v); print(k, "e7 %.3f  bisect(best-match) %.3f  bisect+assign %.3f" % tuple(v.mean(0)))
