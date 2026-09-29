"""Label-free block estimators.  Every function takes only (P, embeddings/features, links, subject ids) -> p1 = p(B1)
per window.  Same code is used for OOF (L0 links) and test (L2 links)."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import lightgbm as lgb
from common import M1, M2, BLK


def mass(P):
    return P[:, M1].sum(1), P[:, M2].sum(1)


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)


def residual_emb(emb, P, sbj):
    """per subject: emb minus its soft class-centroid reconstruction -> mostly scene / person / noise"""
    E = l2n(emb.astype(np.float32)); R = np.empty_like(E)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); Pi = P[ii].astype(np.float32)
        M = (Pi.T @ E[ii]) / (Pi.sum(0)[:, None] + 1e-6)
        R[ii] = E[ii] - Pi @ M
    return R


def centred(X, sbj):
    X = X.astype(np.float32).copy()
    for s in np.unique(sbj):
        ii = sbj == s; X[ii] -= X[ii].mean(0)
    return X


def pca(X, d):
    X = X - X.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (X @ Vt[:d].T).astype(np.float32)


def kmeans(X, k, seed=0, iters=60, restarts=4):
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for _ in range(restarts):
        c = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
            c.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(c)
        for _ in range(iters):
            D = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
            lab = D.argmin(1)
            Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inert = D[np.arange(len(X)), lab].sum()
        if inert < best[0]:
            best = (inert, lab)
    return best[1]


def cluster_p1(lab, e, f, prior=1.0):
    """cluster-level block posterior = B1 mass fraction of the cluster (with a weak 0.5 prior)"""
    out = np.empty(len(lab))
    for j in np.unique(lab):
        m = lab == j; out[m] = (e[m].sum() + prior) / (e[m].sum() + f[m].sum() + 2 * prior)
    return out


def knn_graph(X, k):
    """dense within-group kNN (cosine), symmetric binary-ish weights"""
    E = l2n(X); S = E @ E.T; np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, k, axis=1)[:, :k]; n = len(E)
    W = sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))
    return W.maximum(W.T).tocsr()


def row_norm(W):
    d = np.asarray(W.sum(1)).ravel(); d[d == 0] = 1; return (sp.diags(1 / d) @ W).tocsr()


def link_graph(succ, score, n, T=1.0, b=0.0, min_score=None):
    m = succ >= 0
    if min_score is not None:
        m &= score >= min_score
    src, dst = np.flatnonzero(m), succ[m]
    w = 1 / (1 + np.exp(-np.clip((score[m] - b) / T, -20, 20)))
    return sp.csr_matrix((np.r_[w, w], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)).tocsr()


def propagate(F0, A, alpha, iters, self_out=True):
    """F <- alpha*A F + (1-alpha) F0; if not self_out return neighbour-only estimate A F"""
    F = F0.copy()
    for _ in range(iters):
        F = alpha * (A @ F) + (1 - alpha) * F0
    return F if self_out else A @ F


def p1_from_F(F, prior=1e-3):
    return (F[:, 0] + prior) / (F[:, 0] + F[:, 1] + 2 * prior)


# ------------------------------------------------------------------ methods
def m_self(P, **kw):
    e, f = mass(P); return (e + 1e-6) / (e + f + 2e-6)


def m_kmeans(P, emb, sbj, k=2, feat="resid", d=24, **kw):
    e, f = mass(P); out = np.empty(len(P))
    X = residual_emb(emb, P, sbj) if feat == "resid" else centred(l2n(emb), sbj)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); Z = pca(X[ii], d)
        Z = Z / (Z.std(0, keepdims=True) + 1e-6) if feat == "resid_w" else Z
        lab = kmeans(Z, k, seed=int(s))
        out[ii] = cluster_p1(lab, e[ii], f[ii])
    return out


def m_spectral(P, emb, sbj, knn=10, feat="resid", nvec=1, **kw):
    """split on the Fiedler vector of the within-subject kNN graph (sign), cluster-level p1"""
    e, f = mass(P); out = np.empty(len(P))
    X = residual_emb(emb, P, sbj) if feat == "resid" else l2n(emb)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); W = knn_graph(X[ii], knn)
        d = np.asarray(W.sum(1)).ravel(); Dm = sp.diags(1 / np.sqrt(d)); A = Dm @ W @ Dm
        vals, vecs = eigsh(A, k=2 + nvec, which="LA")
        o = np.argsort(-vals); v = (Dm @ vecs[:, o[1]])
        lab = (v > np.median(v)).astype(int)
        out[ii] = cluster_p1(lab, e[ii], f[ii])
    return out


def m_prop(P, emb, sbj, succ, score, graph="link", alpha=0.99, iters=200, knn=10, feat="raw", self_out=False,
           w_link=0.5, min_score=None, F0=None, **kw):
    e, f = mass(P); n = len(P)
    F0 = np.c_[e, f] if F0 is None else F0
    parts = []
    if graph in ("link", "both"):
        parts.append(("link", row_norm(link_graph(succ, score, n, min_score=min_score))))
    if graph in ("knn", "both"):
        X = residual_emb(emb, P, sbj) if feat == "resid" else l2n(emb)
        blocks = []; idx = []
        rows, cols, vals = [], [], []
        for s in np.unique(sbj):
            ii = np.flatnonzero(sbj == s); W = knn_graph(X[ii], knn).tocoo()
            rows.append(ii[W.row]); cols.append(ii[W.col]); vals.append(W.data)
        Wk = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        parts.append(("knn", row_norm(Wk)))
    if len(parts) == 1:
        A = parts[0][1]
    else:
        L, K = parts[0][1], parts[1][1]
        has = np.asarray(L.sum(1)).ravel() > 0
        wl = np.where(has, w_link, 0.0)
        A = (sp.diags(wl) @ L + sp.diags(1 - wl) @ K).tocsr()
    F = propagate(F0, A, alpha, iters, self_out=self_out)
    return p1_from_F(F)


def m_nullscene(P, emb, sbj, succ, score, seed_p1=None, conf=0.3, d=40, lam=0.1, null_thr=0.5, pc_on="null",
                **kw):
    """session-scene direction learned on (predicted) NULL frames, where posture is roughly constant:
    null rows get block pseudo-labels from a propagated estimate (seed_p1, e.g. link/kNN propagation of the activity
    block mass), a per-subject ridge on subject-centred PCA-d video features is fitted on the confident ones
    (2-way cross-fitted) and applied to every row; Platt-calibrated on the pseudo-labels."""
    E = l2n(emb.astype(np.float32)); pn = P[:, 0]; out = np.full(len(P), 0.5)
    rng = np.random.default_rng(0)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); Xc = E[ii] - E[ii].mean(0)
        isn = pn[ii] >= null_thr
        base = Xc[isn] if pc_on == "null" else Xc
        U, S, Vt = np.linalg.svd(base - base.mean(0), full_matrices=False)
        Z = (Xc - base.mean(0)) @ Vt[:d].T; Z /= Z[isn].std(0) + 1e-6
        q = seed_p1[ii]; tr = isn & (np.abs(q - 0.5) >= conf)
        t = np.where(q > 0.5, 1.0, -1.0)
        if tr.sum() < 40 or len(np.unique(t[tr])) < 2:
            continue
        fo = rng.integers(0, 2, len(ii)); sc = np.zeros(len(ii))
        for k in range(2):
            m = tr & (fo != k); A = Z[m]; w = np.where(t[m] > 0, 0.5 / (t[m] > 0).mean(), 0.5 / (t[m] < 0).mean())
            beta = np.linalg.solve((A * w[:, None]).T @ A + lam * len(A) * np.eye(d), (A * w[:, None]).T @ t[m])
            b0 = np.average(t[m] - A @ beta, weights=w)
            sc[fo == k] = Z[fo == k] @ beta + b0
        # Platt scaling on the pseudo-labelled rows (1-D logistic by Newton)
        xs, ys = sc[tr], (t[tr] > 0).astype(float); ab = np.array([1.0, 0.0])
        for _ in range(30):
            z = np.clip(ab[0] * xs + ab[1], -30, 30); p = 1 / (1 + np.exp(-z)); g = np.array([((p - ys) * xs).sum(), (p - ys).sum()])
            wv = p * (1 - p) + 1e-9; H = np.array([[(wv * xs * xs).sum(), (wv * xs).sum()], [(wv * xs).sum(), wv.sum()]]) + 1e-6 * np.eye(2)
            ab -= np.linalg.solve(H, g)
        out[ii] = 1 / (1 + np.exp(-np.clip(ab[0] * sc + ab[1], -30, 30)))
    return out


def m_selftrain(P, emb, feats, sbj, conf=0.8, featset="video", lofo=False, rounds=40, seed=0, **kw):
    """per subject LightGBM block ~ features, trained on high-confidence predicted-activity windows (pseudo-labels),
    5-way cross-fitted; lofo=True: train on the two other exercise families only (isolates scene signal)"""
    fam = np.zeros(19, int); fam[1:6] = 1; fam[6:11] = 2; fam[11:19] = 3
    e, f = mass(P); out = m_self(P).copy()
    cls = P.argmax(1); pmax = P.max(1)
    R = residual_emb(emb, P, sbj)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.5,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=seed)
    rng = np.random.default_rng(seed)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        Xs = [pca(centred(l2n(emb[ii]), np.zeros(len(ii), int)), 32), pca(R[ii], 32)]
        if featset in ("video", "all"):
            Xs += [feats["vpca"][ii], feats["vmot"][ii]]
        if featset == "all":
            Xs.append(np.nan_to_num(feats["imu"][ii]))
        X = np.concatenate(Xs, 1).astype(np.float32)
        tr = (cls[ii] > 0) & (pmax[ii] >= conf); yb = (BLK[cls[ii]] == 1).astype(int)
        famw = fam[cls[ii]]
        p = np.full(len(ii), np.nan)
        if lofo:
            for fm in (1, 2, 3):
                trm = tr & (famw != fm) & (famw > 0); te = famw == fm
                if trm.sum() < 50 or len(np.unique(yb[trm])) < 2:
                    continue
                bst = lgb.train(params, lgb.Dataset(X[trm], yb[trm]), rounds); p[te] = bst.predict(X[te])
            # null windows (argmax 0): average of the three family models is not available -> use all-family model
            te = famw == 0
            if te.any() and tr.sum() >= 50:
                bst = lgb.train(params, lgb.Dataset(X[tr], yb[tr]), rounds); p[te] = bst.predict(X[te])
        else:
            fo = rng.integers(0, 5, len(ii))
            for k in range(5):
                trm = tr & (fo != k); te = fo == k
                bst = lgb.train(params, lgb.Dataset(X[trm], yb[trm]), rounds); p[te] = bst.predict(X[te])
        ok = ~np.isnan(p); out[ii[ok]] = p[ok]
    return out
