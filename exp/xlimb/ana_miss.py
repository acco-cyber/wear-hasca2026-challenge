"""How often is the true successor missing from the video candidates, and where does it rank under cross-limb IMU residuals?"""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
FD = os.path.join(XD, "feats")
xl = pickle.load(open(os.path.join(XD, "xl_models.pkl"), "rb")); meta, imu, vid, sl = load_prep()

def xl_dist(W, limb, xl):
    n = len(W); D = window_desc(W); Xin = np.nan_to_num(np.concatenate([D, raw_tail(W)], 1)); Y = np.nan_to_num(np.concatenate([D, raw_head(W)], 1))
    R2 = np.full((n, n), np.inf, np.float32)
    for a in range(4):
        ia = np.where(limb == a)[0]
        for b in range(4):
            ib = np.where(limb == b)[0]; pred, rs = xl.predict("next", a, b, Xin[ia]); my = xl.m[("next", a, b)][2]
            Pn = pred / rs; Yn = Y[ib] / rs; r2 = (Pn ** 2).sum(1)[:, None] + (Yn ** 2).sum(1)[None] - 2 * Pn @ Yn.T
            b2 = (((Y[ib] - my) / rs) ** 2).sum(1)[None]            # unconditional residual of the destination
            R2[np.ix_(ia, ib)] = (r2 - b2) / Y.shape[1]
    np.fill_diagonal(R2, np.inf); return R2

for s in ["sbj_0", "sbj_5", "sbj_2", "sbj_6"]:
    d = np.load(os.path.join(FD, f"{s}_0.npz")); cand, limb, y = d["cand"], d["limb"], d["y"]; n = len(y); a, b = sl[s]
    W = session_windows(imu, a, b, limb); R2 = xl_dist(W, limb, xl)
    true = np.arange(1, n); inC = np.array([true[i - 0] in cand[i] for i in range(n - 1)])
    rk = np.array([(R2[i] < R2[i, i + 1]).sum() for i in range(n - 1)])
    same = limb[:-1] == limb[1:]
    print(s, f"miss {1 - inC.mean():.3f} | rank of true succ under xl: all median {np.median(rk):.0f}, top10 {np.mean(rk < 10):.3f} top20 {np.mean(rk < 20):.3f}"
          f" | among missing: top10 {np.mean(rk[~inC] < 10):.3f} top20 {np.mean(rk[~inC] < 20):.3f} | same-limb top10 {np.mean(rk[same] < 10):.3f} cross top10 {np.mean(rk[~same] < 10):.3f}"
          f" | missing & same-label-as-pred? y-change frac among missing {np.mean(y[:-1][~inC] != y[1:][~inC]):.3f}", flush=True)
