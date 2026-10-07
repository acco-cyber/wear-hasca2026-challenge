"""(b) label-free session-block posterior by long-range diffusion of the predicted block over the link graph.
For a tile predicted class c the evidence excludes every tile predicted c (otherwise a wrong-block bout votes for
itself).  Diagnostic: how often does the posterior point to the TRUE session block on (i) the wrong-block tiles we
would like to fix and (ii) correct activity tiles we must not break.  Graphs: estimated links (8 matchings), +video kNN,
and the oracle true_succ chain as an upper bound."""
from common import *
import scipy.sparse as sp

d = base(); y = d["oof_y"]; tb = true_block(d); rec = d["oof_rec"]; t = d["t"]
lk = np.load(os.path.join(K7, "links.npz")); su = lk["oof_succ"]; sco = lk["oof_score"]
lab = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labo.npy")).astype(int)
E = np.load(os.path.join(K7, "oof_emb.npy")).astype(np.float32)
art = ((rec == 3) & (t >= 2564)) | ((rec == 14) & (t >= 3445))


def graph(ii, kind, knn=10):
    n = len(ii); loc = -np.ones(len(y), int); loc[ii] = np.arange(n); rows, cols, w = [], [], []
    if kind in ("links", "links+knn"):
        for k in range(su.shape[0]):
            s_ = su[k, ii]; ok = s_ >= 0; a = np.flatnonzero(ok); b = loc[s_[ok]]; g = b >= 0
            rows += [a[g]]; cols += [b[g]]; w += [np.ones(g.sum()) / su.shape[0]]
    if kind == "true":
        s_ = d["true_succ"][ii]; ok = s_ >= 0; a = np.flatnonzero(ok); b = loc[s_[ok]]; g = b >= 0
        rows += [a[g]]; cols += [b[g]]; w += [np.ones(g.sum())]
    if kind in ("knn", "links+knn"):
        Z = E[ii] - E[ii].mean(0); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        for a0 in range(0, n, 2000):
            S = Z[a0:a0 + 2000] @ Z.T
            for j in range(S.shape[0]):
                S[j, a0 + j] = -9
            nb = np.argpartition(-S, knn, axis=1)[:, :knn]
            rows += [np.repeat(np.arange(a0, a0 + S.shape[0]), knn)]; cols += [nb.ravel()]; w += [np.full(nb.size, 0.5 / knn)]
    A = sp.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    A = A + A.T; dg = np.asarray(A.sum(1)).ravel() + 1e-9
    return sp.diags(1 / dg) @ A                          # row-stochastic


def diffuse(Pm, Y, alpha, iters=300):
    F = Y.copy()
    for _ in range(iters):
        F = alpha * (Pm @ F) + (1 - alpha) * Y
    return F


def posterior(lab_, kind, alpha):
    """pi[i] = P(session block of tile i == A) from the evidence of tiles NOT predicted lab_[i]"""
    pi = np.full(len(y), 0.5)
    for r in np.unique(rec):
        ii = ordered(d, r); Pm = graph(ii, kind); L = lab_[ii]
        for c in range(1, 19):
            m = L == c
            if not m.any():
                continue
            Y = np.zeros((len(ii), 2)); act = (L > 0) & (L != c)
            Y[act, BLK[L[act]] - 1] = 1
            F = diffuse(Pm, Y, alpha); pi[ii[m]] = (F[m, 0] + 1e-6) / (F[m].sum(1) + 2e-6)
    return pi


if __name__ == "__main__":
    bp = BLK[lab]; act = (bp > 0) & ~art
    wrong = act & (bp != tb); right = act & (bp == tb)
    print(f"activity-predicted tiles {act.sum()}, wrong session block {wrong.sum()} (true null {(wrong & (y == 0)).sum()})")
    res = {}
    for kind in ("true", "links", "knn", "links+knn"):
        for alpha in (0.9, 0.99):
            pi = posterior(lab, kind, alpha); res[(kind, alpha)] = pi
            pt = np.where(tb == 1, pi, 1 - pi)            # posterior of the TRUE block
            pp = np.where(bp == 1, pi, 1 - pi)            # posterior of the PREDICTED block
            print(f"{kind:10s} a={alpha}: true-block acc (pi>.5) all-act {(pt[act] > .5).mean():.3f} | on wrong-block tiles: "
                  f"P(true blk)>.5 {(pt[wrong] > .5).mean():.3f} >.9 {(pt[wrong] > .9).mean():.3f} | on right-block tiles: "
                  f"P(pred blk)<.5 {(pp[right] < .5).mean():.3f} <.1 {(pp[right] < .1).mean():.3f}", flush=True)
    np.save(os.path.join(OUT, "s07_pi.npy"), np.array([res], dtype=object), allow_pickle=True)
