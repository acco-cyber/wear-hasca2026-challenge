"""Test-side link fusion: 4-limb sample continuity (from the grouped 2025 windows) + the L2 video link, Hungarian per
subject -> fused successor links for our 12,234 tiles.
  python fuse_test.py <groups.npz> <out_links.npz> [--lam 6] [--k_extra 25] [--w2 1.0] [--l2 path]
groups.npz: w25_rows (12234,4) row index into work/w25/w25.npz for each limb of the tile's second (-1 unknown)."""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import KEEP
W = r"E:\Claude code\wear"; LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
ap = argparse.ArgumentParser(); ap.add_argument("groups"); ap.add_argument("out"); ap.add_argument("--lam", type=float, default=6.0)
ap.add_argument("--k_extra", type=int, default=25); ap.add_argument("--w2", type=float, default=1.0); ap.add_argument("--l2", default=os.path.join(KEEP, "links_L2_test.npz"))
ap.add_argument("--miss_w", type=float, default=0.0, help="cost weight for limbs whose window is unknown (0 = ignore those channels)")
a = ap.parse_args()
g = np.load(a.groups); rows = g["w25_rows"].astype(np.int64)
w25 = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = w25["acc"].astype(np.float32)
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float32)
tm = pd.read_csv(os.path.join(W, "data", "test", "test_meta_data.csv")); sbj = tm.sbj_id.to_numpy(); limb = np.array([LIMBS.index(x) for x in tm.sensor_location])
N = len(A26)
# 4-limb tail/head per tile: own limb from the tile itself, others from the grouped 2025 rows
tail = np.zeros((N, 4, 3), np.float32); tail2 = np.zeros((N, 4, 3), np.float32); head = np.zeros((N, 4, 3), np.float32); head2 = np.zeros((N, 4, 3), np.float32)
known = np.zeros((N, 4), bool)
for L in range(4):
    r = rows[:, L]; m = r >= 0
    tail[m, L] = A25[r[m], -1]; tail2[m, L] = A25[r[m], -2]; head[m, L] = A25[r[m], 0]; head2[m, L] = A25[r[m], 1]; known[m, L] = True
own = np.arange(N)
tail[own, limb] = A26[:, -1]; tail2[own, limb] = A26[:, -2]; head[own, limb] = A26[:, 0]; head2[own, limb] = A26[:, 1]; known[own, limb] = True
print("known limb windows per tile:", np.bincount(known.sum(1), minlength=5).tolist())
l2 = np.load(a.l2); lsucc = l2["succ"].astype(np.int64); lsc = l2["score_qn"].astype(np.float32)
SUCC = np.full(N, -1, np.int64); SCORE = np.full(N, -50.0, np.float32)


def cont(i, j):
    """continuity cost over the limbs known for BOTH rows (i tails, j heads), normalised by the number of such limbs"""
    kk = known[i] & known[j]                                        # (p,4)
    d1 = ((tail[i] - head[j]) ** 2).sum(2); d2 = (((2 * tail[i] - tail2[i]) - head[j]) ** 2).sum(2); d3 = ((tail[i] - (2 * head[j] - head2[j])) ** 2).sum(2)
    c = (d1 + a.w2 * d2 + a.w2 * d3) * kk; nk = kk.sum(1)
    out = c.sum(1) * (4.0 / np.maximum(nk, 1))
    if (nk == 0).any():                                             # no limb in common: neutral cost
        out[nk == 0] = np.median(out[nk > 0]) if (nk > 0).any() else 0.0
    return out


for s in np.unique(sbj):
    ii = np.flatnonzero(sbj == s); n = len(ii); pos = np.full(N, -1); pos[ii] = np.arange(n)
    # candidates: L2 successor + k continuity-nearest heads (per tile, within the subject)
    T = tail[ii].reshape(n, -1); H = head[ii].reshape(n, -1); K = known[ii].astype(np.float32)
    # approximate nearest heads using only limbs known in the tail row (mask), chunked
    cand = np.full((n, a.k_extra + 1), -1, np.int64); lo = np.full((n, a.k_extra + 1), -2.0, np.float32)
    for c0 in range(0, n, 512):
        sl = slice(c0, min(n, c0 + 512)); t = tail[ii[sl]]; kt = known[ii[sl]]
        D = (((t[:, None, :, :] - head[ii][None, :, :, :]) ** 2).sum(3) * (kt[:, None, :] & known[ii][None, :, :])).sum(2)
        D[np.arange(sl.stop - sl.start), np.arange(sl.start, sl.stop)] = np.inf
        cand[sl, 1:] = np.argsort(D, 1)[:, :a.k_extra]
    ls = lsucc[ii]; cand[:, 0] = np.where(ls >= 0, pos[np.maximum(ls, 0)], -1); lo[:, 0] = np.where(ls >= 0, lsc[ii], -50.0)
    ok = cand >= 0; rws = np.repeat(np.arange(n), cand.shape[1]).reshape(cand.shape)
    cost = np.full(cand.shape, np.inf, np.float32); cost[ok] = cont(ii[rws[ok]], ii[cand[ok]])
    fused = np.where(ok, lo - a.lam * np.log1p(10 * cost), -1e6)
    M = np.full((n, n), -1e6, np.float32); M[rws[ok], cand[ok]] = np.maximum(M[rws[ok], cand[ok]], fused[ok]); np.fill_diagonal(M, -1e7)
    r_, c_ = linear_sum_assignment(-M); succ = np.full(n, -1); succ[r_] = c_; succ[M[r_, c_] < -1e5] = -1
    SUCC[ii] = np.where(succ >= 0, ii[np.maximum(succ, 0)], -1); SCORE[ii] = np.where(succ >= 0, M[np.arange(n), np.maximum(succ, 0)], -50.0)
    agree = np.mean((succ >= 0) & (succ == cand[:, 0]))
    print(f"sbj {s}: n={n} linked {np.mean(succ >= 0):.3f} agrees with L2 link {agree:.3f}", flush=True)
m = SUCC >= 0; ref = np.sort(lsc[lsucc >= 0]); rk = np.argsort(np.argsort(SCORE[m])); sq = np.full(N, -50.0, np.float32); sq[m] = ref[(rk * len(ref) // max(m.sum(), 1)).clip(0, len(ref) - 1)]
np.savez(a.out, succ=SUCC, score=SCORE, score_qn=sq); print("saved", a.out, "linked", m.mean().round(3))
