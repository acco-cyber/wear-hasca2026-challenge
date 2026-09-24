"""Potts-MRF refinement (ICM with random partial synchronous updates) on a symmetric neighbour graph, combined with
count calibration. Starts from the chain-Viterbi solution."""
import numpy as np, scipy.sparse as sp
from nbvit import viterbi, logT_matrix, NC

def sym_graph(W):
    W = W.tocsr(); return (W + W.T).tocsr()

def icm(logE, W, lab, lam=1.0, iters=10, frac=0.5, rng=None):
    rng = np.random.RandomState(0) if rng is None else rng; n = len(lab)
    for _ in range(iters):
        O = np.zeros((n, NC)); O[np.arange(n), lab] = 1
        sc = logE + lam * (W @ O); new = sc.argmax(1)
        upd = rng.rand(n) < frac; lab = np.where(upd, new, lab)
    return lab

def seg_refine(logP, W, lab, lam=3.0, lo=85, hi=160, mu=2.0, min_w=0.0, time_limit=20.0, null_free=True):
    """Block move: segments = connected components of same-label edges (weight>min_w) of the decoded labels; re-assign
    every segment to one class by ILP (unary = sum log-prob + lam * votes from outside-segment neighbours under the
    current labels), soft count bounds [lo,hi] per activity class (penalty mu per window)."""
    from scipy.sparse.csgraph import connected_components
    from segdec import seg_ilp_cost
    n = len(lab); Wc = W.tocoo(); m = (lab[Wc.row] == lab[Wc.col]) & (Wc.data > min_w)
    G = sp.csr_matrix((np.ones(m.sum()), (Wc.row[m], Wc.col[m])), shape=(n, n))
    seg = connected_components(G, directed=False)[1]
    O = np.zeros((n, NC)); O[np.arange(n), lab] = 1
    # votes from neighbours outside the own segment
    same = seg[Wc.row] == seg[Wc.col]
    Wout = sp.csr_matrix((Wc.data * (~same), (Wc.row, Wc.col)), shape=(n, n))
    U = logP + lam * (Wout @ O)
    return seg_ilp_cost(U, seg, lo, hi, mu, time_limit), seg

def calib_mrf(logP, packed, W, lo=85, hi=160, p_stay=0.8, lam=1.0, icm_iters=10, iters=40, step=0.25, null_lo=None, null_hi=None):
    b = np.zeros(NC); logT = logT_matrix(p_stay)
    lo_a = np.broadcast_to(np.asarray(lo, float), (NC - 1,)); hi_a = np.broadcast_to(np.asarray(hi, float), (NC - 1,))
    for it in range(iters):
        Z = logP + b[None]; Z = np.maximum(Z - np.logaddexp.reduce(Z, 1, keepdims=True), np.log(1e-6))
        lab = viterbi(Z, packed, logT)
        if lam > 0: lab = icm(Z, W, lab, lam, icm_iters)
        cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo_a)[0] + 1; over = np.where(cnt[1:] > hi_a)[0] + 1
        if len(under) == 0 and len(over) == 0: break
        b[under] += step; b[over] -= step
    return lab, b
