"""Diagnostics: (1) baseline F1; (2) oracle bout-level seriation (true bouts -> mean emb -> order); (3) Mantel-type
distance-vs-|dt| correlation among null windows; (4) supervised slow-drift projection (SFA-like, LOSO) + seriation."""
import sys, time
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from o_common import *
from scipy.ndimage import uniform_filter1d
import hanbat_stack as hs

d = load_meta(); rec, t, y, sbj, fold = d["rec"], d["t"], d["y"], d["sbj"], d["fold"]
P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy")
pred = hs.finish(P, dict(sbj=sbj, sets={0: 2, 14: 2}))
print("baseline F1 %.4f" % hs.macro_f1(y, pred), "per fold", [round(hs.macro_f1(y[fold == f], pred[fold == f]), 4) for f in range(5)])
E = l2n(load_emb("oof"))
Xc = E.copy()
for s in np.unique(sbj):
    ii = sbj == s; Xc[ii] -= Xc[ii].mean(0)
recs = np.unique(rec)
rng = np.random.default_rng(0)

# global class means (for residualising in training sessions) - LOSO computed per target subject below
# per-rec stats for SFA
mu_all = np.stack([Xc[y == c].mean(0) for c in range(N_CLS)])
stats = {}
for r in recs:
    ii = np.flatnonzero(rec == r); o = ii[np.argsort(t[ii])]
    stats[r] = o
out = {"bout_rho_raw": [], "bout_rho_sfa": [], "mantel_null": [], "mantel_all_raw": [], "sfa_spec": [], "sfa_pc1": [],
       "sfa_spec_m5": [], "bout_mantel_raw": []}


def spec_dense(Z, sig=None):
    D = np.sqrt(np.maximum(((Z[:, None] - Z[None]) ** 2).sum(-1), 0))
    sig = np.median(D[D > 0]) if sig is None else sig
    Wd = np.exp(-(D / sig) ** 2); np.fill_diagonal(Wd, 0)
    f, _ = fiedler(Wd, eps=0, nvec=1)
    return f, D


for r in recs:
    ii = stats[r]; s = sbj[ii[0]]; tt = t[ii]; yy = y[ii]
    # LOSO class means
    oth = sbj != s
    mu = np.stack([Xc[oth & (y == c)].mean(0) for c in range(N_CLS)])
    # ---- SFA-like: fit on other recs
    Ss = np.zeros((768, 768)); St = np.zeros((768, 768)); n_ = 0
    for r2 in recs:
        if sbj[stats[r2][0]] == s:
            continue
        jj = stats[r2]
        R = Xc[jj] - mu[y[jj]]                      # remove global class effect
        slow = uniform_filter1d(R, size=301, axis=0, mode="nearest")
        slow = slow - slow.mean(0)
        Xa = Xc[jj] - Xc[jj].mean(0)
        Ss += slow.T @ slow; St += Xa.T @ Xa; n_ += len(jj)
    St += 1e-3 * np.trace(St) / 768 * np.eye(768)
    L = np.linalg.cholesky(St); Li = np.linalg.inv(L)
    ev, evec = np.linalg.eigh(Li @ Ss @ Li.T)
    Wsfa = Li.T @ evec[:, ::-1][:, :20]
    Z = Xc[ii] @ Wsfa
    Z = (Z - Z.mean(0)) / Z.std(0)
    W_, _ = knn_graph(Z[:, :10], k=10, metric="euc")
    f, _ = fiedler_sp(W_)
    out["sfa_spec"].append((r, *eval_order(f, tt)))
    W_, _ = knn_graph(Z[:, :5], k=10, metric="euc")
    f, _ = fiedler_sp(W_)
    out["sfa_spec_m5"].append((r, *eval_order(f, tt)))
    out["sfa_pc1"].append((r, *eval_order(Z[:, 0], tt)))
    # ---- oracle bouts
    chg = np.flatnonzero(np.diff(yy) != 0) + 1
    segs = np.split(np.arange(len(ii)), chg)
    segs = [g for g in segs if len(g) >= 10]
    mid = np.array([tt[g].mean() for g in segs])
    Bm = np.stack([Xc[ii[g]].mean(0) for g in segs])
    f, D = spec_dense(Bm)
    out["bout_rho_raw"].append((r, *eval_order(f, mid)))
    iu = np.triu_indices(len(segs), 1)
    out["bout_mantel_raw"].append((r, spearmanr(D[iu], np.abs(mid[:, None] - mid[None])[iu]).correlation, 0, 0))
    Bz = np.stack([Z[g].mean(0) for g in segs])
    f, _ = spec_dense(Bz[:, :10])
    out["bout_rho_sfa"].append((r, *eval_order(f, mid)))
    # ---- Mantel null windows
    nl = np.flatnonzero(yy == 0)
    a = rng.choice(nl, 20000); b = rng.choice(nl, 20000); k_ = a != b
    dist = 1 - (E[ii[a]] * E[ii[b]]).sum(1)
    out["mantel_null"].append((r, spearmanr(dist[k_], np.abs(tt[a] - tt[b])[k_]).correlation, 0, 0))
    a = rng.integers(0, len(ii), 20000); b = rng.integers(0, len(ii), 20000); k_ = a != b
    dist = 1 - (E[ii[a]] * E[ii[b]]).sum(1)
    out["mantel_all_raw"].append((r, spearmanr(dist[k_], np.abs(tt[a] - tt[b])[k_]).correlation, 0, 0))
    print(r, {k: round(v[-1][1], 3) for k, v in out.items()}, flush=True)

print("\n%-18s %8s %8s %8s %8s %8s" % ("diag", "med|rho|", "min|rho|", "max|rho|", "err%", "err_s"))
for k, L in out.items():
    A = np.array(L, float)
    print("%-18s %8.3f %8.3f %8.3f %8.1f %8.0f" % (k, np.median(np.abs(A[:, 1])), np.abs(A[:, 1]).min(),
                                                np.abs(A[:, 1]).max(), 100 * np.median(A[:, 2]), np.median(A[:, 3])))
