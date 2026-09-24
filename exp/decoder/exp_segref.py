"""MRF decode followed by segment-level ILP block moves (+ optional ICM polish)."""
import sys, time, numpy as np, pandas as pd, scipy.sparse as sp
from common import *
from nbvit import pack
from mrf import calib_mrf, seg_refine, icm
from harness import data
oof, S = data(); SESS = [s for s in EVAL + EXTRA if s in S]
lam = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
rows = []
for s in SESS:
    st = S[s]; P = sess_P(oof, st); y = st["y"]; n = st["n"]
    W0 = graph_matrix(st["cand"], st["lo"], n, k=10); logP = np.log(np.clip(smooth(P, W0, 0.5, 5), 1e-6, 1))
    W = W0 + W0.T; d = np.asarray(W.sum(1)).ravel(); W = (sp.diags(1 / np.where(d > 0, d, 1)) @ W).tocsr()
    t0 = time.time(); lab, b = calib_mrf(logP, pack(base_chains(st)), W, 85, 160, 0.8, lam, 10); r = dict(session=s, mrf=mf1(y, lab))
    Z = logP - np.logaddexp.reduce(logP, 1, keepdims=True)
    for mu in (1.0, 3.0):
        for minw in (0.0, 0.05):
            t1 = time.time(); l2, seg = seg_refine(Z, W, lab, lam, 85, 160, mu, minw)
            if l2 is None: r[f"seg_mu{mu}_w{minw}"] = np.nan; continue
            r[f"seg_mu{mu}_w{minw}"] = mf1(y, l2)
            l3 = icm(Z + b[None], W, l2, lam, 10); r[f"seg_mu{mu}_w{minw}_icm"] = mf1(y, l3)
            print(s, mu, minw, "nseg", seg.max() + 1, "time %.1f" % (time.time() - t1), flush=True)
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows).set_index("session").T
df["EVAL"] = df[EVAL].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(df.round(4).to_string()); df.to_csv(f"exp_segref_lam{lam}.csv")
