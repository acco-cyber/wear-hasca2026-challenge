"""Idea 1: over-segment each subject by k-means in [video mp, sqrt(smoothed posterior)] space, then assign clusters to
classes with an ILP with soft per-class count bounds; optional final chain-Viterbi pass anchored on the ILP labels."""
import sys, time, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from common import *
from segdec import seg_ilp, seg_stats
from nbvit import pack, calib, logT_matrix, viterbi
from vidknn import mp_desc, train_vid
from harness import data

oof, S = data(); SESS = [s for s in EVAL + EXTRA if s in S]
rows = []
for s in SESS:
    st = S[s]; P = sess_P(oof, st); y = st["y"]; n = st["n"]
    logPg = np.log(np.clip(smooth(P, graph_matrix(st["cand"], st["lo"], n, k=10), 0.5, 5), 1e-6, 1))
    pk = pack(base_chains(st)); D = mp_desc(train_vid()[st["a"]:st["b"]])
    r = dict(session=s)
    r["b85"] = mf1(y, calib(logPg, pk, 85, 160, p_stay=0.8)[0])
    for K in (100, 200, 400):
        for wp in (0.5, 1.0, 2.0):
            X = np.c_[D, wp * np.sqrt(np.exp(logPg))]
            seg = KMeans(K, n_init=1, random_state=0).fit_predict(X)
            if wp == 1.0: print(s, K, seg_stats(seg, y), flush=True)
            lab = seg_ilp(logPg, seg, lo=85, hi=160, mu=2.0, time_limit=30)
            r[f"K{K}_wp{wp}"] = mf1(y, lab)
            # anchored Viterbi: emission + gamma * onehot(ILP label), recalibrated
            A = np.zeros((n, NC)); A[np.arange(n), lab] = 1.0
            r[f"K{K}_wp{wp}_anch1"] = mf1(y, calib(logPg + 1.0 * A, pk, 85, 160, p_stay=0.8)[0])
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows).set_index("session").T
df["EVAL"] = df[EVAL].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(df.round(4).to_string()); df.to_csv("exp_kilp.csv")
