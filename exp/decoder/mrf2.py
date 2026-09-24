"""Generalised calibrated MRF: per-class Potts weights (null vs activity), seeds, synchronous fraction, seed voting."""
import numpy as np, scipy.sparse as sp
from nbvit import viterbi, logT_matrix, NC

def icm2(logE, W, lab, lamv, iters=10, frac=0.5, rng=None):
    rng = np.random.RandomState(0) if rng is None else rng; n = len(lab)
    for _ in range(iters):
        O = np.zeros((n, NC)); O[np.arange(n), lab] = 1
        sc = logE + (W @ O) * lamv[None]; new = sc.argmax(1)
        upd = rng.rand(n) < frac; lab = np.where(upd, new, lab)
    return lab

def calib_mrf2(logP, packed, W, lo=80, hi=250, p_stay=0.7, lam=4.0, lam_null=None, icm_iters=10, iters=40, step=0.25,
               frac=0.5, seed=0, polish=0):
    b = np.zeros(NC); logT = logT_matrix(p_stay); lamv = np.full(NC, lam)
    if lam_null is not None: lamv[0] = lam_null
    lo_a = np.broadcast_to(np.asarray(lo, float), (NC - 1,)); hi_a = np.broadcast_to(np.asarray(hi, float), (NC - 1,))
    for it in range(iters):
        Z = logP + b[None]; Z = np.maximum(Z - np.logaddexp.reduce(Z, 1, keepdims=True), np.log(1e-6))
        lab = viterbi(Z, packed, logT)
        lab = icm2(Z, W, lab, lamv, icm_iters, frac, np.random.RandomState(seed))
        cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo_a)[0] + 1; over = np.where(cnt[1:] > hi_a)[0] + 1
        if len(under) == 0 and len(over) == 0: break
        b[under] += step; b[over] -= step
    if polish > 0: lab = icm2(Z, W, lab, lamv, polish, 1.0, np.random.RandomState(seed + 100))
    return lab, b
