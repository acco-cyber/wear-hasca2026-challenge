"""Ceiling test on the Hanbat OOF rows: if the four limb windows of every second were known (perfect grouping), how
exact do successor links get when 4-limb sample continuity is fused with the L2 link candidates?
  python fuse_ceiling.py [--lam 1.0] [--recall_only]
Uses work/hanbat/l2oof/oof_L2.npz (top_cand (N,3), top_lo (N,3), true_succ) and raw train CSVs for all 4 limbs."""
import os, sys, argparse, glob
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import KEEP, HB
W = r"E:\Claude code\wear"
COLS = [f"{l}_acc_{a}" for l in ("left_arm", "left_leg", "right_arm", "right_leg") for a in "xyz"]
ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--recall_only", action="store_true")
ap.add_argument("--k_extra", type=int, default=0, help="also add the k best continuity candidates per row")
ap.add_argument("--save", default="", help="save fused links to work/hanbat/l2oof/<name>.npz (succ, score, global rows)")
ap.add_argument("--w2", type=float, default=0.5, help="weight of the extrapolation terms")
ap.add_argument("--l2_mode", default="top3", choices=["top3", "succ"], help="succ = only the assigned L2 link with its qn score (test condition)")
ap.add_argument("--group_noise", type=float, default=0.0, help="fraction of non-own limb windows replaced by a wrong second's window")
ap.add_argument("--noise_kind", default="bout", choices=["bout", "any"], help="bout: wrong second with the same label; any: random second")
ap.add_argument("--w3", type=float, default=0.0, help="weight of a 3-sample (acceleration-trend) extrapolation term")
ap.add_argument("--agg", default="sum", choices=["sum", "soft", "min2", "med"])
ap.add_argument("--known_frac", type=float, default=1.0, help="fraction of non-own limb windows that are known (others unknown, not wrong)")
a = ap.parse_args()
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
l2 = np.load(os.path.join(HB, "l2oof", "oof_L2.npz")); tc, tl, ts = l2["top_cand"].astype(np.int64), l2["top_lo"].astype(np.float32), l2["true_succ"].astype(np.int64)
if a.l2_mode == "succ":
    tc = l2["succ"].astype(np.int64)[:, None]; tl = l2["score_qn"].astype(np.float32)[:, None]
has = ts >= 0
print("L2 top-k recall of the true successor:", [round(float(np.mean((tc[has, :k] == ts[has, None]).any(1))), 3) for k in (1, 2, 3)],
      "| assigned-link exact", round(float(np.mean(l2["succ"][has] == ts[has])), 3))
if a.recall_only:
    sys.exit()
stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(W, "data", "train", "inertial_feat", "sbj_*.csv")))
N = len(sm["y"]); tail = np.zeros((N, 12), np.float32); tail2 = np.zeros((N, 12), np.float32); head = np.zeros((N, 12), np.float32); head2 = np.zeros((N, 12), np.float32)
for r, stem in enumerate(stems):
    ii = np.flatnonzero(sm["rec"] == r)
    if not len(ii):
        continue
    A = pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{stem}.csv"), usecols=COLS)[COLS].to_numpy(np.float32)
    A = np.nan_to_num(A); st = sm["start"][ii]
    tail[ii] = A[st + 49]; tail2[ii] = A[st + 48]; head[ii] = A[st]; head2[ii] = A[st + 1]
    print("loaded", stem, len(ii), flush=True)
tail3 = np.zeros_like(tail); head3 = np.zeros_like(head)
if a.w3 > 0:
    for r, stem in enumerate(stems):
        ii = np.flatnonzero(sm["rec"] == r)
        if len(ii):
            A = np.nan_to_num(pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{stem}.csv"), usecols=COLS)[COLS].to_numpy(np.float32)); st = sm["start"][ii]
            tail3[ii] = A[st + 47]; head3[ii] = A[st + 2]
if a.group_noise > 0:                                   # corrupt the grouping: replace non-own limb windows (12-ch layout: limb-major)
    rng = np.random.default_rng(0); own = sm["sensor"]  # their sensor order ra,rl,ll,la -> our limb order la,ll,ra,rl
    own_limb = np.array([3, 1, 2, 0])[own]  # placeholder mapping fixed below
    S2L = {0: 2, 1: 3, 2: 1, 3: 0}; own_limb = np.array([S2L[int(s)] for s in own])
    for L in range(4):
        sl = slice(3 * L, 3 * L + 3); m = (own_limb != L) & (rng.random(N) < a.group_noise)
        src = np.arange(N)
        for r in np.unique(sm["rec"]):
            ii = np.flatnonzero((sm["rec"] == r) & m)
            if not len(ii):
                continue
            pool = np.flatnonzero(sm["rec"] == r)
            if a.noise_kind == "bout":
                for i in ii:
                    same = pool[sm["y"][pool] == sm["y"][i]]; src[i] = rng.choice(same)
            else:
                src[ii] = rng.choice(pool, len(ii))
        for arr in (tail, tail2, head, head2, tail3, head3):
            arr[m, sl] = arr[src[m], sl]


KNOWN = np.ones((N, 4), bool)
if a.known_frac < 1.0:
    rng_k = np.random.default_rng(1); S2L = {0: 2, 1: 3, 2: 1, 3: 0}; own_limb = np.array([S2L[int(s)] for s in sm["sensor"]])
    KNOWN = rng_k.random((N, 4)) < a.known_frac; KNOWN[np.arange(N), own_limb] = True
    print("known limbs per row:", np.bincount(KNOWN.sum(1), minlength=5).tolist())


def cont_cost(i, j):
    """continuity cost between the tail of rows i and the head of rows j, per limb then aggregated (--agg)"""
    def per_limb(x):
        return x.reshape(len(x), 4, 3).sum(2)
    d1 = per_limb((tail[i] - head[j]) ** 2); d2 = per_limb(((2 * tail[i] - tail2[i]) - head[j]) ** 2); d3 = per_limb((tail[i] - (2 * head[j] - head2[j])) ** 2)
    c = d1 + a.w2 * d2 + a.w2 * d3                       # (p,4)
    if a.w3 > 0:
        d4 = per_limb(((3 * tail[i] - 3 * tail2[i] + tail3[i]) - head[j]) ** 2); d5 = per_limb((tail[i] - (3 * head[j] - 3 * head2[j] + head3[j])) ** 2)
        c = c + a.w3 * (d4 + d5)
    if a.known_frac < 1.0:
        kk = KNOWN[i] & KNOWN[j]; c = c * kk; nk = kk.sum(1)
        out = c.sum(1) * (4.0 / np.maximum(nk, 1))
        if (nk == 0).any():                               # no limb in common: neutral cost (median of the informative pairs)
            out[nk == 0] = np.median(out[nk > 0]) if (nk > 0).any() else 0.0
        return out
    if a.agg == "sum":
        return c.sum(1)
    if a.agg == "soft":                                    # heavy-tailed: each limb saturates, a wrong limb cannot veto
        return np.log1p(10 * c).sum(1) / 4
    if a.agg == "min2":
        return np.sort(c, 1)[:, :2].sum(1)
    if a.agg == "med":
        return np.median(c, 1)
    raise ValueError(a.agg)


# per recording: candidate set = L2 top-3 (+ optional k continuity-nearest), fused score = lo - lam * log1p(cost*10); Hungarian
exact_fused, exact_l2, n_tot = 0, 0, 0
SUCC = np.full(N, -1, np.int64); SCORE = np.full(N, -50.0, np.float32)
for r in np.unique(sm["rec"]):
    ii = np.flatnonzero(sm["rec"] == r); pos = np.full(N, -1); pos[ii] = np.arange(len(ii)); n = len(ii)
    cand = tc[ii]; lo = tl[ii].copy(); ok = cand >= 0
    if a.k_extra > 0:                                  # continuity-nearest heads for every tail (within the recording)
        C = ((tail[ii][:, None, :] - head[ii][None, :, :]) ** 2).sum(2); np.fill_diagonal(C, np.inf)
        extra = np.argsort(C, 1)[:, :a.k_extra]; cand = np.concatenate([cand, ii[extra]], 1); lo = np.concatenate([lo, np.full(extra.shape, -2.0, np.float32)], 1); ok = cand >= 0
    rows = np.repeat(np.arange(n), cand.shape[1]).reshape(cand.shape)
    cost = np.full(cand.shape, np.inf, np.float32)
    cc = cont_cost(ii[rows[ok]], cand[ok]); cost[ok] = cc
    fused = np.where(ok, lo - a.lam * (cost if a.agg == "soft" else np.log1p(10 * cost)), -1e6)
    M = np.full((n, n), -1e6, np.float32); M[rows[ok], pos[cand[ok]]] = fused[ok]; np.fill_diagonal(M, -1e7)
    rr, cc_ = linear_sum_assignment(-M); succ = np.full(n, -1); succ[rr] = cc_; succ[M[rr, cc_] < -1e5] = -1
    SUCC[ii] = np.where(succ >= 0, ii[np.maximum(succ, 0)], -1); SCORE[ii] = np.where(succ >= 0, M[np.arange(n), np.maximum(succ, 0)], -50.0)
    tloc = np.where(ts[ii] >= 0, pos[np.maximum(ts[ii], 0)], -1); h = tloc >= 0
    exact_fused += int(np.sum(succ[h] == tloc[h])); exact_l2 += int(np.sum(pos[np.maximum(l2["succ"][ii][h], 0)] * (l2["succ"][ii][h] >= 0) + -1 * (l2["succ"][ii][h] < 0) == tloc[h])); n_tot += int(h.sum())
print(f"lam {a.lam} k_extra {a.k_extra} w2 {a.w2}: exact L2 {exact_l2 / n_tot:.3f} -> fused with perfect-group 4-limb continuity {exact_fused / n_tot:.3f}")
if a.save:
    # score for the graph stage: map fused score to a log-odds-like range (quantiles of the L2 qn scores)
    m = SUCC >= 0; ref = np.sort(l2["score_qn"][l2["succ"] >= 0]); rk = np.argsort(np.argsort(SCORE[m])); sq = ref[(rk * len(ref) // max(m.sum(), 1)).clip(0, len(ref) - 1)]
    out = np.full(N, -50.0, np.float32); out[m] = sq
    np.savez(os.path.join(HB, "l2oof", a.save + ".npz"), succ=SUCC, score=SCORE, score_qn=out, rows=np.arange(N)); print("saved", a.save)
