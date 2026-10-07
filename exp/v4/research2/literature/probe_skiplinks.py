"""Train simulation of '2025-data skip links' (literature line, cheap, label-free method; labels only for scoring).
Per recording and limb: naive continuity Hungarian over ALL 1-s tiles of that limb (what the 2nd-challenge test.csv
offers for subjects 22-25). Then pick one random limb per second (the 2026 test design) and, for every picked tile,
walk its limb chain forward up to K hops until it reaches another picked tile -> a same-limb 'skip link' with a
claimed offset k. Reports coverage, exact rate (lands exactly k seconds later), same-label rate, and the same for
confident subsets (all hops with cost-margin above a quantile).
  python probe_skiplinks.py sbj_5 sbj_9 ...
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W = r"E:\Claude code\wear"
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
subs = sys.argv[1:] or ["sbj_5", "sbj_9", "sbj_12", "sbj_20", "sbj_2", "sbj_21"]
K = 6
rng = np.random.default_rng(0)


def cost_matrix(T):
    a49, a48 = T[:, 49], T[:, 48]
    b0, b1 = T[:, 0], T[:, 1]
    fa = a49 + (a49 - a48); bb = b0 - (b1 - b0)
    sa = np.abs(np.diff(T[:, -6:], axis=1)).mean((1, 2)) + 2e-3
    sb = np.abs(np.diff(T[:, :6], axis=1)).mean((1, 2)) + 2e-3
    d1 = ((fa[:, None, :] - b0[None, :, :]) ** 2).sum(-1)
    d2 = ((a49[:, None, :] - bb[None, :, :]) ** 2).sum(-1)
    C = (d1 + d2) / ((sa[:, None] + sb[None, :]) ** 2)
    np.fill_diagonal(C, 1e9)
    return C


rec = []
for s in subs:
    df = pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{s}.csv"))
    n = len(df) // 50
    lab = df["label"].fillna("null").astype(str).to_numpy()[: n * 50].reshape(n, 50)[:, 25]
    succ = np.full((4, n), -1); conf = np.zeros((4, n))
    for li, limb in enumerate(LIMBS):
        X = np.nan_to_num(df[[f"{limb}_acc_{a}" for a in "xyz"]].to_numpy(np.float32))[: n * 50].reshape(n, 50, 3)
        C = cost_matrix(X)
        r, c = linear_sum_assignment(C)
        succ[li, r] = c
        # confidence: log ratio of second-best to assigned cost in the row (assignment-aware margin)
        Cr = C[r].copy(); asg = Cr[np.arange(len(r)), c]; Cr[np.arange(len(r)), c] = np.inf
        conf[li, r] = np.log((Cr.min(1) + 1e-6) / (asg + 1e-6))
    pick = rng.integers(0, 4, n)            # 2026 design: one random limb per second
    for rep in range(1):
        for t in range(n):
            li = pick[t]; cur = t; ok_conf = np.inf
            for k in range(1, K + 1):
                nxt = succ[li, cur]
                if nxt < 0:
                    break
                ok_conf = min(ok_conf, conf[li, cur])
                cur = nxt
                if pick[cur] == li:
                    rec.append(dict(sbj=s, limb=li, k=k, exact=int(cur == t + k), same=int(lab[cur] == lab[t]),
                                    near=int(abs(cur - t) <= 10), conf=ok_conf, base_same=int(t + k < n and lab[min(t + k, n - 1)] == lab[t])))
                    break
    print(s, n, "tiles; skip links so far", len(rec), flush=True)
R = pd.DataFrame(rec)
ntiles = sum(len(pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{s}.csv"), usecols=["sbj_id"])) // 50 for s in subs)
print(f"\ncoverage (picked tiles with a skip link within {K} hops): {len(R) / ntiles:.3f}")
print("by hop count k:\n", R.groupby("k")[["exact", "same", "near", "base_same"]].agg(["mean", "count"]).round(3))
for q in (0.0, 0.25, 0.5, 0.75):
    thr = R.conf.quantile(q); S = R[R.conf >= thr]
    print(f"conf >= q{q:.2f}: kept {len(S) / ntiles:.3f} of tiles | exact {S.exact.mean():.3f} same-label {S.same.mean():.3f} within10s {S.near.mean():.3f} | oracle same-label at true offset {S.base_same.mean():.3f}")
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_skiplinks.csv"), index=False)
