"""Seriation test: does a spectral embedding of the link graph recover coarse time within a session?"""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\exp\decoder")
from tlib import load_structs
from common import graph_matrix
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
S = {}
for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
for s in list(R)[:18]:
    st = S[s]; n = st["n"]; y = R[s]["y"]; t = np.arange(n)
    out = []
    for k in (1, 3, 10):
        W = graph_matrix(st["cand"], st["lo"], n, k=k); W = W + W.T
        d = np.asarray(W.sum(1)).ravel() + 1e-9; Dm = sp.diags(1 / np.sqrt(d)); Nm = Dm @ W @ Dm
        vals, vecs = eigsh(Nm, k=6, which="LA")
        o = np.argsort(-vals); vecs = vecs[:, o] / np.sqrt(d)[:, None]
        rhos = [abs(spearmanr(vecs[:, j], t).correlation) for j in range(1, 6)]
        # block membership recoverable? best |rho| of eigvecs with true block (A=1)
        out.append(f"k{k}: max|rho t| {max(rhos):.2f} (ev {int(np.argmax(rhos))+1})")
    print(s, n, " ".join(out), flush=True)
