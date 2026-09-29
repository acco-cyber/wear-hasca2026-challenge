"""Task 1: label-free seriation of OOF windows per recording. Prints Spearman(order, t) and median position error."""
import sys, time
from o_common import *

d = load_meta(); rec, t, y, sbj, fold = d["rec"], d["t"], d["y"], d["sbj"], d["fold"]
E = l2n(load_emb("oof"))
# subject-centred
Xc = E.copy()
for s in np.unique(sbj):
    ii = sbj == s; Xc[ii] -= Xc[ii].mean(0)
links = np.load(K + r"\links_L0.npz"); succ = links["oof_succ"]; lsc = links["oof_score"]
recs = np.unique(rec)
res = {}


def add(name, r, score, signed=False):
    ii = np.flatnonzero(rec == r)
    rho, ef, es = eval_order(score, t[ii], signed)
    res.setdefault(name, []).append((r, rho, ef, es, t[ii].max() + 1))


def proj_out(X, U):
    return X - (X @ U) @ U.T


t0 = time.time()
for r in recs:
    ii = np.flatnonzero(rec == r); s = sbj[ii[0]]
    oth = np.flatnonzero(sbj != s)
    X = Xc[ii]
    # --- diagnostics: kNN temporal locality
    W, nb = knn_graph(X, k=10)
    dt = np.abs(t[ii][nb] - t[ii][:, None])
    res.setdefault("diag_knn10_dt", []).append((r, np.median(dt), (dt <= 30).mean(), (dt <= 100).mean(),
                                                 (y[ii][nb] == y[ii][:, None]).mean()))
    # (a) spectral seriation
    for k in (10, 30):
        W, _ = knn_graph(X, k=k)
        f, _ = fiedler_sp(W)
        add(f"a_spec_k{k}", r, f)
    # (b) project out activity directions fit on other subjects
    Xo = Xc[oth]; yo = y[oth]; so = sbj[oth]
    mu = np.stack([Xo[yo == c].mean(0) for c in range(N_CLS)])
    U, S_, Vt = np.linalg.svd(mu - mu.mean(0), full_matrices=False)
    B = Vt[:18].T
    # per-subject class-mean deviations -> many between-class directions
    devs = []
    for s2 in np.unique(so):
        jj = so == s2
        for c in range(N_CLS):
            m_ = jj & (yo == c)
            if m_.sum() >= 10:
                devs.append(Xo[m_].mean(0))
    devs = np.stack(devs); devs -= devs.mean(0)
    _, _, Vd = np.linalg.svd(devs, full_matrices=False)
    # LDA
    Sw = np.zeros((768, 768))
    for c in range(N_CLS):
        Z = Xo[yo == c] - mu[c]; Sw += Z.T @ Z
    Sw /= len(Xo); Sw += 1e-3 * np.trace(Sw) / 768 * np.eye(768)
    Sb = (mu - mu.mean(0)).T @ (mu - mu.mean(0)) / N_CLS
    L = np.linalg.cholesky(Sw); Li = np.linalg.inv(L)
    ev, evec = np.linalg.eigh(Li @ Sb @ Li.T)
    Wl = Li.T @ evec[:, ::-1][:, :18]
    Ql, _ = np.linalg.qr(Wl)
    variants = {"b_cm18": B, "b_lda18": Ql, "b_dev50": Vd[:50].T, "b_dev100": Vd[:100].T, "b_dev200": Vd[:200].T,
                "b_lda18+dev100": np.linalg.qr(np.hstack([Ql, Vd[:100].T]))[0]}
    for nm, Uq in variants.items():
        Xr = proj_out(X, Uq)
        W, _ = knn_graph(Xr, k=10)
        f, _ = fiedler_sp(W)
        add(nm + "_spec", r, f)
        # also first PC of residual (linear drift)
        Xr0 = Xr - Xr.mean(0)
        u, s_, vt = np.linalg.svd(Xr0, full_matrices=False)
        add(nm + "_pc1", r, u[:, 0])
    # (c) successor-link graph (+ weak kNN)
    loc = -np.ones(len(E), int); loc[ii] = np.arange(len(ii))
    sv = succ[ii]; ok = (sv >= 0) & (loc[np.maximum(sv, 0)] >= 0)
    a = np.flatnonzero(ok); b = loc[sv[ok]]
    Wlink = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(len(ii), len(ii))); Wlink = Wlink.maximum(Wlink.T)
    f, _ = fiedler_sp(Wlink, eps=1e-2)
    add("c_links", r, f)
    Wk, _ = knn_graph(X, k=10)
    for lam in (0.3, 1.0, 3.0):
        f, _ = fiedler_sp(Wk + lam * Wlink)
        add(f"c_links{lam}+knn10", r, f)
    print(r, "done %.0fs" % (time.time() - t0), flush=True)

# (d) ridge regression of t/T on centred embedding, trained on other subjects (signed)
for alpha in (1.0, 10.0, 100.0):
    for r in recs:
        ii = np.flatnonzero(rec == r); s = sbj[ii[0]]
        oth = np.flatnonzero(sbj != s)
        tt = np.empty(len(oth))
        for r2 in np.unique(rec[oth]):
            jj = rec[oth] == r2; tt[jj] = t[oth][jj] / t[oth][jj].max()
        Xo = Xc[oth]
        w = np.linalg.solve(Xo.T @ Xo + alpha * np.eye(768), Xo.T @ (tt - tt.mean()))
        add(f"d_ridge{alpha}", r, Xc[ii] @ w, signed=True)

print("\n%-22s %7s %7s %7s %7s %7s %7s" % ("method", "med|rho|", "min", "max", "medRho", "err%", "err_s"))
for nm, L in res.items():
    A = np.array(L, float)
    if nm.startswith("diag"):
        print(nm, "median over recs: med|dt| %.0f  frac<=30 %.2f  <=100 %.2f  same-label %.2f" % tuple(np.median(A[:, 1:], 0)))
        continue
    print("%-22s %7.3f %7.3f %7.3f %7.3f %7.1f %7.0f" % (nm, np.median(np.abs(A[:, 1])), np.abs(A[:, 1]).min(),
                                                      np.abs(A[:, 1]).max(), np.median(A[:, 1]), 100 * np.median(A[:, 2]),
                                                      np.median(A[:, 3])))
np.save(OUT + r"\s1_res.npy", np.array([(nm, *row) for nm, L in res.items() for row in L], dtype=object), allow_pickle=True)
