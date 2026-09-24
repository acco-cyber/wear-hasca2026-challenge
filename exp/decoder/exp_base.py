"""Reproduce the baseline decoder on the eval sessions and print diagnostics."""
import time, numpy as np, pandas as pd
from common import *
from decode import build_graph, graph_smooth

t0 = time.time(); oof = load_oof(0.2); S = load_structs(("eval",)); print("loaded", time.time() - t0, flush=True)
rows = []
for s in EVAL:
    st = S[s]; P = sess_P(oof, st); y = st["y"]; n = st["n"]
    t1 = time.time(); lab = baseline(P, st); t2 = time.time()
    # check sparse smoothing equals the reference implementation
    Pg_ref = graph_smooth(P, build_graph(st["cand"], st["lo"], n, k=10), 0.5, 5)
    Pg = smooth(P, graph_matrix(st["cand"], st["lo"], n, k=10), 0.5, 5)
    cnt = np.bincount(lab, minlength=NC)
    rows.append(dict(session=s, n=n, null_true=(y == 0).mean(), null_pred=(lab == 0).mean(), raw=mf1(y, P.argmax(1)), base=mf1(y, lab),
                     smooth_maxdiff=np.abs(Pg - Pg_ref).max(), argmax_agree=(Pg.argmax(1) == Pg_ref.argmax(1)).mean(), sec=t2 - t1))
    print(rows[-1], flush=True)
    print("  per-class F1:", np.round(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y, lab, average=None, labels=np.arange(NC)), 2).tolist())
df = pd.DataFrame(rows); print(df.round(4).to_string(index=False)); print(df.mean(numeric_only=True).round(4).to_dict())
