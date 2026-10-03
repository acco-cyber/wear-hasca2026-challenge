"""Calibrated link fusion for partially known limbs: per-limb continuity costs -> learned log-likelihood ratio (true vs
hard-negative candidate pairs), summed over the limbs known for both windows, plus the L2 video link log-odds.
  python fuse_lr.py cv  [--known_frac f] [--group_noise q] [--k_extra 25] [--w_video 1.0] [--save name]
  (fits the LR table on the OOF rows with perfect groups and writes work/hanbat/l2oof/lr_table.npz for the test script)"""
import os, sys, argparse, glob
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import KEEP, HB
W = r"E:\Claude code\wear"
COLS = [f"{l}_acc_{a}" for l in ("left_arm", "left_leg", "right_arm", "right_leg") for a in "xyz"]
S2L = {0: 2, 1: 3, 2: 1, 3: 0}
ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--known_frac", type=float, default=1.0); ap.add_argument("--group_noise", type=float, default=0.0)
ap.add_argument("--k_extra", type=int, default=25); ap.add_argument("--w_video", type=float, default=1.0); ap.add_argument("--prior", type=float, default=-3.0)
ap.add_argument("--save", default=""); ap.add_argument("--w2", type=float, default=1.0); ap.add_argument("--nbins", type=int, default=40)
a = ap.parse_args()
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}; N = len(sm["y"])
l2 = np.load(os.path.join(HB, "l2oof", "oof_L2.npz")); ts = l2["true_succ"].astype(np.int64); lsucc = l2["succ"].astype(np.int64); lsc = l2["score_qn"].astype(np.float32)
stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(W, "data", "train", "inertial_feat", "sbj_*.csv")))
tail = np.zeros((N, 4, 3), np.float32); tail2 = tail.copy(); head = tail.copy(); head2 = tail.copy()
for r, stem in enumerate(stems):
    ii = np.flatnonzero(sm["rec"] == r)
    if not len(ii):
        continue
    A = np.nan_to_num(pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{stem}.csv"), usecols=COLS)[COLS].to_numpy(np.float32)).reshape(-1, 4, 3); st = sm["start"][ii]
    tail[ii] = A[st + 49]; tail2[ii] = A[st + 48]; head[ii] = A[st]; head2[ii] = A[st + 1]
own = np.array([S2L[int(s)] for s in sm["sensor"]]); rng = np.random.default_rng(0)
KNOWN = rng.random((N, 4)) < a.known_frac; KNOWN[np.arange(N), own] = True
if a.group_noise > 0:   # wrong (same-label) windows for a fraction of the known non-own limbs
    for L in range(4):
        m = KNOWN[:, L] & (own != L) & (rng.random(N) < a.group_noise)
        for r in np.unique(sm["rec"]):
            ii = np.flatnonzero((sm["rec"] == r) & m)
            pool = np.flatnonzero(sm["rec"] == r)
            for i in ii:
                j = rng.choice(pool[sm["y"][pool] == sm["y"][i]])
                for arr in (tail, tail2, head, head2):
                    arr[i, L] = arr[j, L]
print("known limbs per row:", np.bincount(KNOWN.sum(1), minlength=5).tolist(), "noise", a.group_noise)


def limb_costs(i, j):
    d1 = ((tail[i] - head[j]) ** 2).sum(2); d2 = (((2 * tail[i] - tail2[i]) - head[j]) ** 2).sum(2); d3 = ((tail[i] - (2 * head[j] - head2[j])) ** 2).sum(2)
    return np.log1p(10 * (d1 + a.w2 * (d2 + d3)))            # (p,4) transformed cost x


# candidates per recording: L2 link + k continuity-nearest heads (distance over commonly known limbs, neutral otherwise)
CAND = np.full((N, a.k_extra + 1), -1, np.int64); ROWI = np.repeat(np.arange(N), a.k_extra + 1).reshape(N, -1)
for r in np.unique(sm["rec"]):
    ii = np.flatnonzero(sm["rec"] == r); n = len(ii)
    T = tail[ii]; H = head[ii]; K = KNOWN[ii]
    D = np.zeros((n, n), np.float32)
    for c0 in range(0, n, 400):
        sl = slice(c0, min(n, c0 + 400)); kk = K[sl][:, None, :] & K[None, :, :]
        d = (((T[sl][:, None] - H[None]) ** 2).sum(3) * kk).sum(2) * (4.0 / np.maximum(kk.sum(2), 1))
        d[kk.sum(2) == 0] = np.inf; D[sl] = d
    np.fill_diagonal(D, np.inf)
    near = np.argsort(D, 1)[:, :a.k_extra]
    cand = np.full((n, a.k_extra + 1), -1, np.int64); cand[:, 1:] = ii[near]
    ls = lsucc[ii]; cand[:, 0] = np.where(ls >= 0, ls, -1)
    CAND[ii] = cand
ok = CAND >= 0
X = np.full(CAND.shape + (4,), np.nan, np.float32); X[ok] = limb_costs(ROWI[ok], CAND[ok])
KK = np.zeros(CAND.shape + (4,), bool); KK[ok] = KNOWN[ROWI[ok]] & KNOWN[CAND[ok]]
is_true = ok & (CAND == ts[ROWI])
# LR table (shared over limbs): bins of x; positives = true pairs, negatives = other candidates (hard negatives)
xs_pos = X[is_true][KK[is_true]]; xs_neg = X[ok & ~is_true][KK[ok & ~is_true]]
edges = np.quantile(np.concatenate([xs_pos, xs_neg]), np.linspace(0, 1, a.nbins + 1)); edges[0], edges[-1] = -np.inf, np.inf
hp = np.histogram(xs_pos, edges)[0] + 1.0; hn = np.histogram(xs_neg, edges)[0] + 1.0
lr = np.log((hp / hp.sum()) / (hn / hn.sum())); lr = np.clip(lr, -6, 6)
print("LR table (per-limb log-likelihood ratio by cost bin):", np.round(lr[[0, 5, 10, 20, 30, -1]], 2).tolist())
if a.save:
    np.savez(os.path.join(HB, "l2oof", "lr_table.npz"), edges=edges[1:-1], lr=lr)


def lr_of(x):
    return lr[np.clip(np.searchsorted(edges[1:-1], x), 0, len(lr) - 1)]


S = np.where(ok, a.prior, -1e6).astype(np.float32)
S[:, 0] = np.where(ok[:, 0], a.w_video * lsc[ROWI[:, 0]], -1e6)
contrib = np.where(KK, lr_of(np.nan_to_num(X)), 0.0).sum(2); S = S + np.where(ok, contrib, 0.0)
# Hungarian per recording
exact = n_tot = 0; SUCC = np.full(N, -1, np.int64); SCORE = np.full(N, -50.0, np.float32)
for r in np.unique(sm["rec"]):
    ii = np.flatnonzero(sm["rec"] == r); n = len(ii); pos = np.full(N, -1); pos[ii] = np.arange(n)
    cand = CAND[ii]; sc = S[ii]; okr = cand >= 0; rows = np.repeat(np.arange(n), cand.shape[1]).reshape(cand.shape)
    M = np.full((n, n), -1e6, np.float32); np.maximum.at(M, (rows[okr], pos[cand[okr]]), sc[okr]); np.fill_diagonal(M, -1e7)
    rr, cc = linear_sum_assignment(-M); succ = np.full(n, -1); succ[rr] = cc; succ[M[rr, cc] < -1e5] = -1
    SUCC[ii] = np.where(succ >= 0, ii[np.maximum(succ, 0)], -1); SCORE[ii] = np.where(succ >= 0, M[np.arange(n), np.maximum(succ, 0)], -50)
    tl = np.where(ts[ii] >= 0, pos[np.maximum(ts[ii], 0)], -1); h = tl >= 0; exact += int(np.sum(succ[h] == tl[h])); n_tot += int(h.sum())
print(f"known_frac {a.known_frac} noise {a.group_noise}: fused (calibrated LR) exact {exact / n_tot:.3f}   [L2 alone 0.532]")
if a.save:
    m = SUCC >= 0; ref = np.sort(lsc[lsucc >= 0]); rk = np.argsort(np.argsort(SCORE[m])); sq = np.full(N, -50.0, np.float32); sq[m] = ref[(rk * len(ref) // max(m.sum(), 1)).clip(0, len(ref) - 1)]
    np.savez(os.path.join(HB, "l2oof", a.save + ".npz"), succ=SUCC, score=SCORE, score_qn=sq, rows=np.arange(N)); print("saved", a.save)
