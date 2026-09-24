"""Sweep null_scale x count band for the baseline decoder on sim sessions (blend OOF)."""
import os, sys, pickle, numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from chain import cut, chains_from_succ
from decode import calibrate_counts, build_graph, graph_smooth
W = r"E:\Claude code\wear\work"
S = pickle.load(open(os.path.join(W, "sim_struct.pkl"), "rb"))
A = np.load(os.path.join(W, "lgbm_v1", "oof.npy")); B = np.load(os.path.join(W, "fusion_v1", "oof.npy"))
oof = np.exp(0.8 * np.log(np.clip(A, 1e-6, 1)) + 0.2 * np.log(np.clip(B, 1e-6, 1))); oof /= np.nansum(oof, 2, keepdims=True)
res = {}
for s, st in S.items():
    n = st["n"]; y = st["y"]
    P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]]; ok = ~np.isnan(P[:, 0]); P = np.where(ok[:, None], P, 1.0 / 19)
    g = build_graph(st["cand"], st["lo"], n, k=10); chains = chains_from_succ(cut(st["succ0"], st["sc"], st["Lm"], -6.0))
    for ns in (0.6, 0.5, 0.4):
        Pn = P.copy(); Pn[:, 0] *= ns; Pn /= Pn.sum(1, keepdims=True); Pg = graph_smooth(Pn, g, alpha=0.5, iters=5)
        for band in ((80, 250), (85, 250), (90, 250), (95, 250), (100, 250), (90, 400)):
            for ps in (0.7, 0.8):
                lab, _ = calibrate_counts(Pg, chains, lo=band[0], hi=band[1], p_stay=ps)
                res.setdefault((ns, band, ps), []).append(f1_score(y, lab, average="macro"))
    print("done", s, flush=True)
df = pd.DataFrame([(k[0], k[1], k[2], *[round(x, 4) for x in v], round(np.mean(v), 4)) for k, v in res.items()],
                  columns=["ns", "band", "ps"] + list(S.keys()) + ["mean"]).sort_values("mean", ascending=False)
pd.set_option("display.width", 250); print(df.head(15).to_string(index=False))
