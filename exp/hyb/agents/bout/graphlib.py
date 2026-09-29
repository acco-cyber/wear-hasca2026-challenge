"""per-subject graph construction + Louvain (numpy/python) + ICM relabelling"""
import numpy as np, scipy.sparse as sp


def knn_topk(E, P, ii, K=40, use_p=0.5):
    S = E[ii] @ E[ii].T
    if use_p:
        Qs = np.sqrt(P[ii]).astype(np.float32); S = S + use_p * (Qs @ Qs.T)
    np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, K, axis=1)[:, :K]; v = np.take_along_axis(S, nb, 1)
    o = np.argsort(-v, 1)
    return np.take_along_axis(nb, o, 1), np.take_along_axis(v, o, 1)


def build_W(nb, vals, k, tau, succ_loc, score, beta, link_b=0.0, mutual=False):
    m = len(nb); rows = np.repeat(np.arange(m), k); vk = vals[:, :k]
    w = np.exp((vk - vk.max()) / tau).reshape(-1)
    Wk = sp.csr_matrix((w, (rows, nb[:, :k].reshape(-1))), shape=(m, m))
    Wk = Wk.minimum(Wk.T) if mutual else Wk.maximum(Wk.T)
    if beta > 0:
        ok = succ_loc >= 0; a = np.flatnonzero(ok); b = succ_loc[ok]
        wl = beta / (1 + np.exp(-(score[ok] - link_b)))
        Wl = sp.csr_matrix((np.r_[wl, wl], (np.r_[a, b], np.r_[b, a])), shape=(m, m))
        Wk = Wk + Wl
    Wk = Wk.tocsr(); Wk.setdiag(0); Wk.eliminate_zeros()
    return Wk


try:
    import numba

    @numba.njit(cache=True)
    def _move(indptr, indices, data, k, m2, gamma, order):
        n = len(k); comm = np.arange(n); tot = k.copy()
        acc = np.zeros(n); touched = np.zeros(n, np.int64)
        moved_any = False
        for sweep in range(30):
            moves = 0
            for i in order:
                ci = comm[i]; nt = 0
                for p in range(indptr[i], indptr[i + 1]):
                    j = indices[p]
                    if j == i:
                        continue
                    c = comm[j]
                    if acc[c] == 0.0:
                        touched[nt] = c; nt += 1
                    acc[c] += data[p]
                tot[ci] -= k[i]
                best_c = ci; best_g = acc[ci] - gamma * tot[ci] * k[i] / m2
                for t in range(nt):
                    c = touched[t]; g = acc[c] - gamma * tot[c] * k[i] / m2
                    if g > best_g + 1e-12:
                        best_g = g; best_c = c
                for t in range(nt):
                    acc[touched[t]] = 0.0
                acc[ci] = 0.0
                comm[i] = best_c; tot[best_c] += k[i]
                if best_c != ci:
                    moves += 1
            if moves == 0:
                break
            moved_any = True
        return comm, moved_any
except Exception:   # pragma: no cover
    _move = None


def louvain_fast(W, gamma=1.0, seed=0, max_levels=10):
    rng = np.random.default_rng(seed)
    n0 = W.shape[0]; g2 = np.arange(n0); A = W.tocsr().astype(np.float64)
    for level in range(max_levels):
        n = A.shape[0]; k = np.asarray(A.sum(1)).ravel(); m2 = k.sum()
        if m2 == 0:
            break
        A.sort_indices()
        comm, moved = _move(A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data, k, m2, float(gamma),
                            rng.permutation(n).astype(np.int64))
        u, comm = np.unique(comm, return_inverse=True); g2 = comm[g2]
        if not moved or len(u) == n:
            break
        M = sp.csr_matrix((np.ones(n), (np.arange(n), comm)), shape=(n, len(u)))
        A = (M.T @ A @ M).tocsr()
    return g2


def louvain(W, gamma=1.0, seed=0, max_levels=10):
    if _move is not None:
        return louvain_fast(W, gamma, seed, max_levels)
    return louvain_slow(W, gamma, seed, max_levels)


def louvain_slow(W, gamma=1.0, seed=0, max_levels=10):
    """returns community label per node"""
    rng = np.random.default_rng(seed)
    n0 = W.shape[0]; node2comm_global = np.arange(n0)
    A = W.tocsr().astype(np.float64)
    for level in range(max_levels):
        n = A.shape[0]; k = np.asarray(A.sum(1)).ravel(); m2 = k.sum()
        if m2 == 0:
            break
        selfw = A.diagonal().copy()
        comm = np.arange(n); tot = k.copy()
        indptr, indices, data = A.indptr, A.indices, A.data
        moved_any = False
        for sweep in range(20):
            moves = 0
            for i in rng.permutation(n):
                a, b = indptr[i], indptr[i + 1]; nbr = indices[a:b]; wts = data[a:b]
                sel = nbr != i; nbr = nbr[sel]; wts = wts[sel]
                ci = comm[i]
                if len(nbr) == 0:
                    continue
                cs = comm[nbr]
                uc, inv = np.unique(cs, return_inverse=True); kin = np.bincount(inv, weights=wts)
                tot[ci] -= k[i]
                kin_own = kin[uc == ci].sum() if (uc == ci).any() else 0.0
                gains = kin - gamma * tot[uc] * k[i] / m2
                g_own = kin_own - gamma * tot[ci] * k[i] / m2
                j = int(np.argmax(gains))
                if gains[j] > g_own + 1e-12 and uc[j] != ci:
                    comm[i] = uc[j]; tot[uc[j]] += k[i]; moves += 1
                else:
                    tot[ci] += k[i]
            if moves == 0:
                break
            moved_any = True
        u, comm = np.unique(comm, return_inverse=True)
        node2comm_global = comm[node2comm_global]
        if not moved_any or len(u) == n:
            break
        # aggregate
        M = sp.csr_matrix((np.ones(n), (np.arange(n), comm)), shape=(n, len(u)))
        A = (M.T @ A @ M).tocsr()
    return node2comm_global


def icm(W, L0, a=1.0, iters=10, init=None):
    """hard-label Potts relabelling: x_i = argmax_c a*L0[i,c] + sum_j W_ij [x_j=c]"""
    n, C = L0.shape; x = L0.argmax(1) if init is None else init.copy()
    Wr = W.tocsr()
    for _ in range(iters):
        OH = np.zeros((n, C)); OH[np.arange(n), x] = 1
        S = a * L0 + Wr @ OH
        xn = S.argmax(1)
        if (xn == x).all():
            break
        x = xn
    return x
