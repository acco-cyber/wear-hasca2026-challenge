"""Batched Viterbi over many chains + count-calibrated decoding with flexible targets."""
import numpy as np
NC = 19

class ChainBatch:
    def __init__(self, chains, n):
        chains = sorted(chains, key=len, reverse=True); L = max(len(c) for c in chains); m = len(chains)
        idx = np.full((m, L), -1, np.int64)
        for i, c in enumerate(chains): idx[i, :len(c)] = c
        self.idx = idx; self.mask = idx >= 0; self.len = self.mask.sum(1); self.n = n; self.L = L
        self.nact = self.mask.sum(0)          # number of active chains at each position (prefix, since sorted)

def logT_matrix(p_stay, nc=NC):
    T = np.full((nc, nc), np.log((1 - p_stay) / (nc - 1))); np.fill_diagonal(T, np.log(p_stay)); return T

def viterbi_batch(logE, cb, logT, edge_bonus=None):
    """logE (n,NC) emission log-scores; cb ChainBatch; logT (NC,NC) or callable. edge_bonus: optional (m,L) array
    scaling of the 'switch' penalty per edge (t-1 -> t). Returns labels (n,)."""
    idx = cb.idx; m, L = idx.shape; na = cb.nact
    delta = logE[idx[:, 0]].copy(); back = [None] * L
    final = np.zeros((m, NC));
    for t in range(1, L):
        k = na[t]
        if k < na[t - 1]: final[k:na[t - 1]] = delta[k:na[t - 1]]     # chains that ended at t-1
        d = delta[:k]
        if edge_bonus is None: M = d[:, :, None] + logT[None]
        else: M = d[:, :, None] + edge_bonus[:k, t][:, None, None] * logT[None]
        b = M.argmax(1); back[t] = b
        delta = np.take_along_axis(M, b[:, None, :], 1)[:, 0] + logE[idx[:k, t]]
    final[:na[L - 1]] = delta
    lab = np.zeros((m, L), np.int64); cur = final.argmax(1); lab[np.arange(m), cb.len - 1] = cur
    for t in range(L - 1, 0, -1):
        k = na[t]; prev = back[t][np.arange(k), cur[:k]]
        cur = cur.copy(); cur[:k] = prev; lab[:k, t - 1] = prev
    out = np.empty(cb.n, np.int64); out[idx[cb.mask]] = lab[cb.mask]
    return out

def calib_decode(logP, cb, lo=60, hi=160, p_stay=0.8, iters=40, step=0.25, b0=None, null_lo=None, null_hi=None,
                 edge_bonus=None, target=None, verbose=False):
    """calibrate_counts re-implementation (same update rule) with optional initial bias, null bounds, per-class
    lo/hi arrays. Returns labels, bias."""
    n = len(logP); b = np.zeros(NC) if b0 is None else b0.copy(); logT = logT_matrix(p_stay)
    lo_a = np.broadcast_to(np.asarray(lo, float), (NC - 1,)); hi_a = np.broadcast_to(np.asarray(hi, float), (NC - 1,))
    for it in range(iters):
        Z = logP + b[None]; Z = np.maximum(Z - np.logaddexp.reduce(Z, 1, keepdims=True), np.log(1e-6))
        lab = viterbi_batch(Z, cb, logT, edge_bonus)
        cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo_a)[0] + 1; over = np.where(cnt[1:] > hi_a)[0] + 1
        nul_u = null_lo is not None and cnt[0] < null_lo; nul_o = null_hi is not None and cnt[0] > null_hi
        if verbose: print(it, cnt.tolist())
        if len(under) == 0 and len(over) == 0 and not nul_u and not nul_o: break
        b[under] += step; b[over] -= step
        if nul_u: b[0] += step
        if nul_o: b[0] -= step
    return lab, b
