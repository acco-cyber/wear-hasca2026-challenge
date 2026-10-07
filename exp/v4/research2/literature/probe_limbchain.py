"""Cheap feasibility probe (literature line): if every second of ONE limb is available as non-overlapping 1-s tiles
(the 2nd-challenge test.csv has all four limbs of every 2026 test second), how well does pure within-limb sample
continuity + Hungarian 1-to-1 assignment recover the successor of each tile?
Simulated on train recordings (tiles at offset 0, one limb at a time). No labels used; no fitting.
  python probe_limbchain.py [subjects...]
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

W = r"E:\Claude code\wear"
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
subs = sys.argv[1:] or ["sbj_5", "sbj_9", "sbj_12", "sbj_20"]


def cost_matrix(T, mode):
    """T: (n,50,3) tiles. returns (n,n) cost for i -> j (j follows i)."""
    a49, a48, a47 = T[:, 49], T[:, 48], T[:, 47]
    b0, b1, b2 = T[:, 0], T[:, 1], T[:, 2]
    if mode == "lin":
        fa = a49 + (a49 - a48)                  # forward prediction of next sample
        bb = b0 - (b1 - b0)                      # backward prediction of previous sample
    else:  # quadratic
        fa = 3 * a49 - 3 * a48 + a47
        bb = 3 * b0 - 3 * b1 + b2
    # noise scale per tile from mean abs first difference (tail of i, head of j)
    sa = np.abs(np.diff(T[:, -6:], axis=1)).mean((1, 2)) + 2e-3
    sb = np.abs(np.diff(T[:, :6], axis=1)).mean((1, 2)) + 2e-3
    d1 = ((fa[:, None, :] - b0[None, :, :]) ** 2).sum(-1)
    d2 = ((a49[:, None, :] - bb[None, :, :]) ** 2).sum(-1)
    C = (d1 + d2) / ((sa[:, None] + sb[None, :]) ** 2)
    np.fill_diagonal(C, 1e9)
    return C


rows = []
for s in subs:
    df = pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{s}.csv"))
    n = len(df) // 50
    for limb in LIMBS:
        X = np.nan_to_num(df[[f"{limb}_acc_{a}" for a in "xyz"]].to_numpy(np.float32))[: n * 50].reshape(n, 50, 3)
        lab = df["label"].astype(str).to_numpy()[: n * 50].reshape(n, 50)[:, 25]
        true = np.arange(1, n + 1); true[-1] = -1
        for mode in ("lin", "quad"):
            t0 = time.time()
            C = cost_matrix(X, mode)
            top1 = C.argmin(1)
            r, c = linear_sum_assignment(C)
            succ = np.full(n, -1); succ[r] = c
            ok = true >= 0
            st = lab == "null"
            # static-ish tiles: low within-tile motion
            mot = np.abs(np.diff(X, axis=1)).mean((1, 2))
            low = mot < np.quantile(mot, 0.25)
            rows.append(dict(sbj=s, limb=limb, mode=mode, n=n,
                             top1=np.mean(top1[ok] == true[ok]),
                             hung=np.mean(succ[ok] == true[ok]),
                             hung_null=np.mean(succ[ok & st] == true[ok & st]),
                             hung_act=np.mean(succ[ok & ~st] == true[ok & ~st]),
                             hung_lowmot=np.mean(succ[ok & low] == true[ok & low]),
                             sec=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
R = pd.DataFrame(rows)
print(R.groupby("mode")[["top1", "hung", "hung_null", "hung_act", "hung_lowmot"]].mean().round(4))
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_limbchain.csv"), index=False)
