"""Within-family variant disambiguation at bout-group level (post-processing of mrf4 labels)."""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

FAMILIES = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12], [13, 14], [16, 17]]

def assign_groups(G, max_per=4, M=1e4, keep=None, keep_w=0.0):
    """G (K,|F|) group utilities. Every group gets one variant, every variant at least one group (if K>=|F|),
    at most max_per groups per variant. Returns variant index per group."""
    K, V = G.shape; copies = max(1, min(max_per, K), int(np.ceil(K / V)))   # every group must get a slot
    cost = np.concatenate([-G - (M if c == 0 else 0.0) for c in range(copies)], 1)   # (K, V*copies)
    r, c = linear_sum_assignment(cost); out = np.zeros(K, np.int64); out[r] = c % V
    return out

def family_pass(lab, logP, groups_fn, fams=FAMILIES, min_win=10, max_per=4, util="fam", temp=1.0):
    lab = lab.copy()
    for F in fams:
        F = np.asarray(F); idx = np.where(np.isin(lab, F))[0]
        if len(idx) < min_win * len(F): continue
        g = groups_fn(idx, F, lab)
        _, g = np.unique(g, return_inverse=True); K = g.max() + 1
        U = logP[idx][:, F] / temp
        if util == "fam": U = U - logsumexp(U, 1, keepdims=True)
        G = np.zeros((K, len(F))); np.add.at(G, g, U)
        v = assign_groups(G, max_per); lab[idx] = F[v[g]]
    return lab

# ---------------- grouping functions ----------------
def groups_oracle(y):
    def f(idx, F, lab): return y[idx]
    return f

def groups_current(idx, F, lab): return lab[idx]

def _sub_affinity(W, idx, vidD=None, kv=0, wv=0.1):
    A = W[idx][:, idx]; A = (A + A.T) * 0.5
    if vidD is not None and kv > 0:
        D = vidD[idx]; S = D @ D.T; np.fill_diagonal(S, -1); k = min(kv, len(idx) - 1)
        nb = np.argpartition(-S, k, axis=1)[:, :k]; r = np.repeat(np.arange(len(idx)), k)
        V = sp.csr_matrix((np.full(len(r), wv), (r, nb.ravel())), shape=A.shape); A = A + (V + V.T) * 0.5
    return A.tocsr()

def groups_spectral(W, extra=0, vidD=None, kv=10, wv=0.05, seed=0):
    from sklearn.cluster import SpectralClustering
    def f(idx, F, lab):
        K = len(F) + extra; A = _sub_affinity(W, idx, vidD, kv, wv)
        A = A + sp.identity(A.shape[0]) * 1e-6
        try:
            return SpectralClustering(K, affinity="precomputed", assign_labels="cluster_qr", random_state=seed).fit_predict(A)
        except Exception:
            return lab[idx]
    return f

def groups_louvain(W, res=1.0, vidD=None, kv=10, wv=0.05, seed=0):
    import networkx as nx
    def f(idx, F, lab):
        A = _sub_affinity(W, idx, vidD, kv, wv).tocoo(); G = nx.Graph()
        G.add_nodes_from(range(len(idx))); G.add_weighted_edges_from(zip(A.row.tolist(), A.col.tolist(), A.data.tolist()))
        comms = nx.community.louvain_communities(G, weight="weight", resolution=res, seed=seed)
        g = np.zeros(len(idx), np.int64)
        for i, c in enumerate(comms): g[list(c)] = i
        return g
    return f

def groups_cc(R, C, Lo, tau=0.0):
    """connected components of same-current-label link edges with log-odds > tau, inside the family windows"""
    from scipy.sparse.csgraph import connected_components
    def f(idx, F, lab):
        n = len(lab); pos = np.full(n, -1); pos[idx] = np.arange(len(idx))
        m = (pos[R] >= 0) & (pos[C] >= 0) & (Lo > tau) & (lab[R] == lab[C])
        A = sp.csr_matrix((np.ones(m.sum()), (pos[R[m]], pos[C[m]])), shape=(len(idx), len(idx)))
        return connected_components(A, directed=False)[1]
    return f

def fam_icm(lab, U, W, lam=8.0, iters=10, fams=FAMILIES, seed=0):
    """re-run ICM restricted to the family's labels for windows currently in the family"""
    lab = lab.copy(); rng = np.random.RandomState(seed); n = len(lab)
    for F in fams:
        F = np.asarray(F); idx = np.where(np.isin(lab, F))[0]
        if len(idx) == 0: continue
        Wf = W[idx][:, idx]
        for _ in range(iters):
            O = np.zeros((len(idx), len(F))); O[np.arange(len(idx)), np.searchsorted(F, lab[idx])] = 1
            sc = U[idx][:, F] + lam * (Wf @ O); new = F[sc.argmax(1)]
            upd = rng.rand(len(idx)) < 0.5; lab[idx] = np.where(upd, new, lab[idx])
    return lab

def groups_split(W, big=150, vidD=None, kv=10, wv=0.05, seed=0):
    """current labels; any label group larger than `big` windows is split in two by spectral clustering."""
    from sklearn.cluster import SpectralClustering
    def f(idx, F, lab):
        g = lab[idx].copy(); nxt = g.max() + 1
        for v in F:
            m = np.where(g == v)[0]
            if len(m) <= big: continue
            A = _sub_affinity(W, idx[m], vidD, kv, wv) + sp.identity(len(m)) * 1e-6
            try: s = SpectralClustering(2, affinity="precomputed", assign_labels="cluster_qr", random_state=seed).fit_predict(A)
            except Exception: continue
            g[m[s == 1]] = nxt; nxt += 1
        return g
    return f
