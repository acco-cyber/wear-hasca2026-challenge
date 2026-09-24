"""Within-block seriation: restrict link graph to one true block's windows; do eigenvectors track time?"""
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
BA = {4, 5, 8, 9, 10, 15, 16, 17, 18}
for s in R:
    st = S[s]; n = st["n"]; y = R[s]["y"]; t = np.arange(n)
    act = y > 0; isA = np.array([v in BA for v in y])
    # time split: fraction of A before
    tA = np.median(t[act & isA]); tB = np.median(t[act & ~isA])
    # split point: maximize separation
    best = None
    for c in range(0, n, 10):
        sc = np.sum(isA[act] == ((t[act] < c) if tA < tB else (t[act] >= c)))
        if best is None or sc > best[0]: best = (sc, c)
    cpt = best[1]; purity = best[0] / act.sum()
    W = graph_matrix(st["cand"], st["lo"], n, k=10); W = (W + W.T).tocsr()
    out = [f"split@{cpt} pur {purity:.2f}"]
    for name, m in (("first", t < cpt), ("second", t >= cpt)):
        idx = np.where(m)[0]; Wb = W[idx][:, idx]
        d = np.asarray(Wb.sum(1)).ravel() + 1e-9; Dm = sp.diags(1 / np.sqrt(d)); Nm = Dm @ Wb @ Dm
        vals, vecs = eigsh(Nm, k=8, which="LA"); o = np.argsort(-vals); vecs = vecs[:, o] / np.sqrt(d)[:, None]
        rhos = [abs(spearmanr(vecs[:, j], t[idx]).correlation) for j in range(1, 8)]
        # activity-window-only rho
        out.append(f"{name}: max|rho| {max(rhos):.2f} (ev{int(np.argmax(rhos))+1})")
    print(s, " ".join(out), flush=True)
