"""Segment-level constrained assignment decoder."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.optimize import milp, LinearConstraint, Bounds
from common import *

def segments_from_labels(lab0, R, C, Lo, n, tau=-4.0):
    """Connected components of link edges (lo>tau) whose endpoints share the initial label."""
    m = (Lo > tau) & (lab0[R] == lab0[C])
    G = sp.csr_matrix((np.ones(m.sum()), (R[m], C[m])), shape=(n, n))
    return connected_components(G, directed=False)[1]

def seg_stats(seg, y):
    ids, inv = np.unique(seg, return_inverse=True); S = len(ids)
    M = np.zeros((S, NC)); np.add.at(M, (inv, y), 1)
    size = M.sum(1); pur = M.max(1).sum() / len(y)
    return dict(n_seg=S, n_seg_ge10=int((size >= 10).sum()), frac_in_ge10=float(size[size >= 10].sum() / len(y)), purity=float(pur))

def seg_ilp_cost(U, seg, lo=85, hi=160, mu=2.0, time_limit=20.0):
    """U (n,NC) per-window utilities (higher = better). Maximise sum over segments of the chosen class utility with
    soft per-activity count bounds. Returns labels (n,)."""
    return seg_ilp(U, seg, lo=lo, hi=hi, mu=mu, time_limit=time_limit)

def seg_ilp(logP, seg, lo=75, hi=130, mu=2.0, null_lo=None, null_hi=None, time_limit=20.0, max_sets=None, set_pen=0.0):
    """Assign each segment one class. Soft window-count bounds per activity class (penalty mu per window outside).
    logP (n,NC) log-probs (already smoothed). Returns labels (n,)."""
    n = len(logP); ids, inv = np.unique(seg, return_inverse=True); S = len(ids)
    Cst = np.zeros((S, NC)); np.add.at(Cst, inv, -logP)
    size = np.bincount(inv, minlength=S).astype(float)
    nx = S * NC; nu = NC - 1
    # variables: x (S*NC, row-major s,k), u (18), v (18)
    c = np.r_[Cst.ravel(), np.full(nu, mu), np.full(nu, mu)]
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    # each segment exactly one class
    for k in range(NC):
        rows.append(np.arange(S)); cols.append(np.arange(S) * NC + k); vals.append(np.ones(S))
    lb += [1.0] * S; ub += [1.0] * S; r = S
    # activity counts: size.x_k + u_k >= lo ; size.x_k - v_k <= hi
    for k in range(1, NC):
        rows.append(np.full(S, r)); cols.append(np.arange(S) * NC + k); vals.append(size)
        rows.append(np.array([r])); cols.append(np.array([nx + k - 1])); vals.append(np.array([1.0])); lb.append(lo); ub.append(np.inf); r += 1
        rows.append(np.full(S, r)); cols.append(np.arange(S) * NC + k); vals.append(size)
        rows.append(np.array([r])); cols.append(np.array([nx + nu + k - 1])); vals.append(np.array([-1.0])); lb.append(-np.inf); ub.append(hi); r += 1
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nx + 2 * nu))
    integ = np.r_[np.ones(nx), np.zeros(2 * nu)]
    bnds = Bounds(np.zeros(nx + 2 * nu), np.r_[np.ones(nx), np.full(2 * nu, np.inf)])
    res = milp(c, integrality=integ, bounds=bnds, constraints=LinearConstraint(A, lb, ub), options=dict(time_limit=time_limit, disp=False))
    if res.x is None: return None
    X = res.x[:nx].reshape(S, NC); segk = X.argmax(1)
    return segk[inv]
