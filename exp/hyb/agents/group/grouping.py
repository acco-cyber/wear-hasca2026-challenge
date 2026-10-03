"""Anchor-centric grouping + confidence features shared by eval_cfg.py (simulation) and run_test.py (test)."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, identity
from common import hungarian

SYM = {(0, 2), (1, 3)}
PARTNER = {0: 2, 2: 0, 1: 3, 3: 1}
CONF_FEATS = ["sc", "s_anchor", "margin_row", "margin_col", "dyn_a", "dyn_m", "s_sym_partner", "s_own_partner", "mutual"]


def knn_matrix(desc, k):
    X = (desc - desc.mean(0)) / (desc.std(0) + 1e-6); n = len(X)
    Dm = cdist(X, X); np.fill_diagonal(Dm, np.inf)
    nn = np.argpartition(Dm, k, axis=1)[:, :k]
    P = csr_matrix((np.full(n * k, 1.0 / (k + 1), np.float32), (np.repeat(np.arange(n), k), nn.ravel())), shape=(n, n))
    return (P + identity(n, dtype=np.float32, format="csr") * (1.0 / (k + 1))).tocsr()


def smooth_scores(S, Pa, Pb, lam):
    return (S + lam * np.asarray(Pa @ np.asarray((Pb @ S.T)).T)).astype(np.float32)


def sc_of(S, a, b, ia, ib):
    return S[(a, b)][ia, ib] if a < b else S[(b, a)][ib, ia]


def block(S, a, b, ia, ib):
    return S[(a, b)][ia][:, ib] if a < b else S[(b, a)][ib][:, ia].T


def anchor_assign(S, anc, anc_row, is_anchor, n_iter=3, refine=0.5, valid=None):
    """S: canonical score dict {(a,b): N_a x N_b} (a<b) in row space. anc[s]: anchor limb of second s, anc_row[s]: its row,
    is_anchor[m]: bool mask over limb-m rows (rows that are reliable anchors). valid[s]: second s has a reliable anchor
    (default all). Seconds without a reliable anchor are not grouped. Candidate rows (non-anchors) may outnumber the
    seconds to fill (rectangular assignment: every valid second gets a row, leftover rows stay unassigned).
    Returns groups (N,4) and raw confidence features (N,4,len(CONF_FEATS))."""
    from scipy.optimize import linear_sum_assignment
    N = len(anc)
    if valid is None:
        valid = np.ones(N, bool)
    groups = np.full((N, 4), -1, dtype=np.int64)
    groups[np.where(valid)[0], anc[valid]] = anc_row[valid]
    feats = np.full((N, 4, len(CONF_FEATS)), np.nan, dtype=np.float32)
    for it in range(n_iter):
        new = groups.copy()
        for m in range(4):
            secs_m = np.where(valid & (anc != m))[0]; rows_m = np.where(~is_anchor[m])[0]
            assert len(rows_m) >= len(secs_m), (len(rows_m), len(secs_m))
            C = np.zeros((len(rows_m), len(secs_m)), dtype=np.float32); A = np.zeros_like(C)
            for L in range(4):
                if L == m: continue
                sel = np.where(anc[secs_m] == L)[0]
                if len(sel) == 0: continue
                A[:, sel] = block(S, m, L, rows_m, anc_row[secs_m[sel]])
                if it > 0 and refine > 0:
                    for K in range(4):
                        if K in (m, L): continue
                        C[:, sel] += refine * block(S, m, K, rows_m, groups[secs_m[sel], K])
            C += A
            r, c = linear_sum_assignment(-C.astype(np.float64))  # every column (second) gets a distinct row
            new[secs_m[c], m] = rows_m[r]
            if it == n_iter - 1:
                sc = C[r, c]
                top2r = np.partition(C, -2, axis=1)[:, -2:] if C.shape[1] >= 2 else np.c_[C.min(1), C.max(1)]
                top2c = np.partition(C, -2, axis=0)[-2:, :]
                second_r = np.where(top2r[r, 1] == sc, top2r[r, 0], top2r[r, 1])
                second_c = np.where(top2c[1, c] == sc, top2c[0, c], top2c[1, c])
                z = np.zeros_like(sc)
                feats[secs_m[c], m, :] = np.stack([sc, A[r, c], sc - second_r, sc - second_c, z, z, z, z,
                                                   ((sc - second_r) > 0) & ((sc - second_c) > 0)], 1)
        groups = new
    return groups, feats


def finish_feats(S, groups, feats, anc, dyn, valid=None):
    """Fill the dyn flags and the symmetric-partner consistency score. dyn[l]: bool per limb-l row."""
    N = len(anc)
    if valid is None:
        valid = np.ones(N, bool)
    for m in range(4):
        sel = np.where(valid & (anc != m))[0]; r_ = groups[sel, m]
        feats[sel, m, 4] = np.array([dyn[anc[s]][s_row] for s, s_row in zip(sel, groups[sel, anc[sel]])])
        feats[sel, m, 5] = dyn[m][r_]
        part = np.array([PARTNER[a] for a in anc[sel]]); prow = groups[sel, part]
        v = np.array([sc_of(S, m, part[i], r_[i], prow[i]) if part[i] != m else np.nan for i in range(len(sel))], dtype=np.float32)
        v[np.isnan(v)] = feats[sel, m, 1][np.isnan(v)]
        feats[sel, m, 6] = v
        # score of the assigned window with ITS OWN symmetric partner's window in the group (anchor score if that is the anchor)
        own = PARTNER[m]; orow = groups[sel, own]
        feats[sel, m, 7] = sc_of(S, m, own, r_, orow)
    return feats


def is_sym_pair(m, a):
    return (min(m, a), max(m, a)) in SYM


def fit_logistic(X, y, l2=1e-2, it=25):
    Xb = np.c_[np.ones(len(X)), X]; w = np.zeros(Xb.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Xb @ w)); Wt = p * (1 - p) + 1e-9
        g = Xb.T @ (p - y) + l2 * w; H = (Xb * Wt[:, None]).T @ Xb + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def predict_logistic(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


def apply_calib(cal, F, sym_mask):
    Fz = (F.astype(np.float64) - cal["mu"]) / cal["sd"]
    conf = np.empty(len(F))
    conf[sym_mask] = predict_logistic(cal["w_sym"], Fz[sym_mask])
    conf[~sym_mask] = predict_logistic(cal["w_cross"], Fz[~sym_mask])
    return conf.astype(np.float32)
