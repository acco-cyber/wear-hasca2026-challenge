"""Numba Viterbi over chains + count-calibrated decoding."""
import numpy as np
from numba import njit
NC = 19

def pack(chains):
    order = np.concatenate([np.asarray(c, np.int64) for c in chains]); lens = np.array([len(c) for c in chains], np.int64)
    starts = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64); return order, starts, lens

@njit(cache=True)
def _vit(logE, order, starts, lens, logT, ew):
    n, K = logE.shape; out = np.empty(n, np.int64)
    for c in range(len(starts)):
        s = starts[c]; L = lens[c]
        if L == 1:
            out[order[s]] = np.argmax(logE[order[s]]); continue
        back = np.empty((L, K), np.int64); delta = logE[order[s]].copy(); nd = np.empty(K)
        for t in range(1, L):
            i = order[s + t]; w = ew[s + t]
            for k in range(K):
                best = -1e300; bj = 0
                for j in range(K):
                    v = delta[j] + (logT[j, k] if j == k else w * logT[j, k])
                    if v > best: best = v; bj = j
                nd[k] = best + logE[i, k]; back[t, k] = bj
            for k in range(K): delta[k] = nd[k]
        cur = np.argmax(delta); out[order[s + L - 1]] = cur
        for t in range(L - 1, 0, -1):
            cur = back[t, cur]; out[order[s + t - 1]] = cur
    return out

def logT_matrix(p_stay, nc=NC, null_switch=None):
    T = np.full((nc, nc), np.log((1 - p_stay) / (nc - 1))); np.fill_diagonal(T, np.log(p_stay))
    if null_switch is not None:     # cheaper activity<->null switches than activity<->activity
        T[1:, 0] = np.log(null_switch); T[0, 1:] = np.log(null_switch)
    return T

def viterbi(logE, packed, logT, ew=None):
    order, starts, lens = packed
    if ew is None: ew = np.ones(len(order))
    return _vit(np.ascontiguousarray(logE, np.float64), order, starts, lens, np.ascontiguousarray(logT, np.float64), ew)

def calib(logP, packed, lo=60, hi=160, p_stay=0.8, iters=40, step=0.25, b0=None, null_lo=None, null_hi=None, ew=None,
          logT=None, clip=True, return_hist=False):
    """Same update rule as decode.calibrate_counts (additive bias +-step on classes outside [lo,hi])."""
    b = np.zeros(NC) if b0 is None else np.array(b0, float); logT = logT_matrix(p_stay) if logT is None else logT
    lo_a = np.broadcast_to(np.asarray(lo, float), (NC - 1,)); hi_a = np.broadcast_to(np.asarray(hi, float), (NC - 1,))
    for it in range(iters):
        Z = logP + b[None]; Z = Z - np.logaddexp.reduce(Z, 1, keepdims=True)
        if clip: Z = np.maximum(Z, np.log(1e-6))
        lab = viterbi(Z, packed, logT, ew); cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo_a)[0] + 1; over = np.where(cnt[1:] > hi_a)[0] + 1
        nu = null_lo is not None and cnt[0] < null_lo; no = null_hi is not None and cnt[0] > null_hi
        if len(under) == 0 and len(over) == 0 and not nu and not no: break
        b[under] += step; b[over] -= step
        if nu: b[0] += step
        if no: b[0] -= step
    return lab, b
