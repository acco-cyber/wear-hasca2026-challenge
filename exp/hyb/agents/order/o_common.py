import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.stats import spearmanr, rankdata
import scipy.sparse as sp
import scipy.sparse.linalg as spl

K = r"E:\Claude code\wear\work\hanbat\keep"
OUT = r"E:\Claude code\wear\exp\hyb\agents\order"
N_CLS = 19
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]
B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]


def load_meta():
    m = np.load(K + r"\sim_meta.npz")
    d = {k: m[k] for k in m.files}
    d["t"] = d["start"] // 50
    return d


def load_emb(which="oof"):
    e = np.load(K + (r"\oof_emb.npy" if which == "oof" else r"\test_emb.npy")).astype(np.float32)
    return e


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def eval_order(score, t, signed=False):
    """score: 1-D estimated position; t: true seconds. Returns rho, median |err| frac, median |err| s (best orientation
    unless signed)."""
    T = t.max() + 1
    rho = spearmanr(score, t).correlation
    r = (rankdata(score) - 1) / (len(score) - 1)
    tt = t / (T - 1)
    if not signed and rho < 0:
        r = 1 - r
    err = np.abs(r - tt)
    return rho, float(np.median(err)), float(np.median(err) * T)


def knn_graph(X, k=10, metric="cos", mutual=False, tau=None):
    """dense kNN (n up to ~6000). Returns symmetric sparse W."""
    n = len(X)
    if metric == "cos":
        Xn = l2n(X)
        S = Xn @ Xn.T
    else:
        sq = (X ** 2).sum(1)
        S = -(sq[:, None] + sq[None] - 2 * X @ X.T)
    np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    vals = np.take_along_axis(S, nb, 1)
    if tau is None:
        w = np.ones_like(vals)
    else:
        w = np.exp((vals - vals.max(1, keepdims=True)) / tau)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((w.ravel(), (rows, nb.ravel())), shape=(n, n))
    if mutual:
        W = W.minimum(W.T)
    else:
        W = W.maximum(W.T)
    return W, nb


def fiedler_sp(W, eps=1e-3, nvec=1):
    """normalized-Laplacian Fiedler vector(s) of W + c*11^T via LinearOperator + eigsh."""
    W = sp.csr_matrix(W, dtype=np.float64)
    n = W.shape[0]
    c = eps * W.sum() / (n * n)
    d = np.asarray(W.sum(1)).ravel() + c * n
    Dm = 1 / np.sqrt(d)

    def mv(x):
        x = np.asarray(x).reshape(n, -1)
        z = Dm[:, None] * x
        out = W @ z + c * z.sum(0, keepdims=True)
        return (Dm[:, None] * out).reshape(x.shape) if x.shape[1] > 1 else (Dm * np.asarray(out).ravel())

    op = spl.LinearOperator((n, n), matvec=mv, dtype=np.float64)
    vals, vecs = spl.eigsh(op, k=nvec + 1, which="LA", tol=1e-6, maxiter=5000)
    idx = np.argsort(-vals)
    v = vecs[:, idx[1:1 + nvec]] * Dm[:, None]
    return (v[:, 0] if nvec == 1 else v), vals[idx]


def fiedler(W, eps=1e-3, nvec=2):
    """normalized-Laplacian Fiedler vector; adds eps-uniform connection for connectivity (via dense if small)."""
    n = W.shape[0]
    Wd = W.toarray() if sp.issparse(W) else W.copy()
    Wd = Wd + eps * Wd.sum() / (n * n)
    d = Wd.sum(1)
    Dm = 1 / np.sqrt(d)
    Ms = Dm[:, None] * Wd * Dm[None]
    vals, vecs = np.linalg.eigh(Ms)
    # largest eigvals of normalized adjacency == smallest of Laplacian
    idx = np.argsort(-vals)
    v = vecs[:, idx[1:1 + nvec]] * Dm[:, None]
    return v[:, 0] if nvec == 1 else v, vals[idx[:1 + nvec]]
