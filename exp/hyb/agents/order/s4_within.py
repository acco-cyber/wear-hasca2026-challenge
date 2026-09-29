"""Within-SESSION (= within protocol block) seriation. Test sessions each hold one 9-exercise block, so the relevant
problem is ordering windows inside one block. Train recs hold both blocks; split each at the B1/B2 boundary (split point
maximising block purity of exercise windows) and seriate each part separately."""
import sys, time
from o_common import *

d = load_meta(); rec, t, y, sbj, fold = d["rec"], d["t"], d["y"], d["sbj"], d["fold"]
E = l2n(load_emb("oof"))
links = np.load(K + r"\links_L0.npz"); succ = links["oof_succ"]
recs = np.unique(rec)
inB1 = np.isin(y, B1); inB2 = np.isin(y, B2)
parts = []  # (rec, idx sorted by t)
for r in recs:
    ii = np.flatnonzero(rec == r); ii = ii[np.argsort(t[ii])]
    a = np.cumsum(inB1[ii]); b = np.cumsum(inB2[ii])
    # first part B1, second B2  vs  first B2, second B1
    s1 = a + (b[-1] - b); s2 = b + (a[-1] - a)
    k1, k2 = s1.argmax(), s2.argmax()
    k, pur = (k1, s1[k1]) if s1[k1] >= s2[k2] else (k2, s2[k2])
    pur = pur / (inB1[ii].sum() + inB2[ii].sum())
    parts.append((r, 0, ii[:k + 1], pur)); parts.append((r, 1, ii[k + 1:], pur))
print("block-split purity per rec:", [round(p[3], 3) for p in parts[::2]])
print("part lengths:", [len(p[2]) for p in parts])

res = {}


def add(nm, key, score, tt):
    res.setdefault(nm, []).append((key, *eval_order(score, tt), tt.max() - tt.min() + 1))


def proj_out(X, U):
    return X - (X @ U) @ U.T


Xs = E.copy()
for s in np.unique(sbj):
    ii = sbj == s; Xs[ii] -= Xs[ii].mean(0)
t0 = time.time()
for (r, h, ii, pur) in parts:
    s = sbj[ii[0]]; tt = t[ii] - t[ii].min()
    X = E[ii] - E[ii].mean(0)                     # session-centred
    oth = np.flatnonzero(sbj != s); Xo = Xs[oth]; yo = y[oth]; so = sbj[oth]
    devs = [Xo[(so == s2) & (yo == c)].mean(0) for s2 in np.unique(so) for c in range(N_CLS)
            if ((so == s2) & (yo == c)).sum() >= 10]
    devs = np.stack(devs); devs -= devs.mean(0)
    Vd = np.linalg.svd(devs, full_matrices=False)[2]
    key = (r, h)
    W, _ = knn_graph(X, k=10); add("a_spec_k10", key, fiedler_sp(W)[0], tt)
    for q in (100, 200):
        Xr = proj_out(X, Vd[:q].T)
        W2, _ = knn_graph(Xr, k=10); add(f"b_dev{q}_spec", key, fiedler_sp(W2)[0], tt)
    loc = -np.ones(len(E), int); loc[ii] = np.arange(len(ii))
    sv = succ[ii]; ok = (sv >= 0) & (loc[np.maximum(sv, 0)] >= 0)
    a_ = np.flatnonzero(ok); b_ = loc[sv[ok]]
    Wl = sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(len(ii), len(ii))); Wl = Wl.maximum(Wl.T)
    add("c_links1+knn10", key, fiedler_sp(W + Wl)[0], tt)
    # oracle bouts inside the part
    yy = y[ii]; chg = np.flatnonzero(np.diff(yy) != 0) + 1
    segs = [g for g in np.split(np.arange(len(ii)), chg) if len(g) >= 10]
    if len(segs) >= 4:
        Bm = np.stack([X[g].mean(0) for g in segs]); mid = np.array([tt[g].mean() for g in segs])
        D = np.sqrt(np.maximum(((Bm[:, None] - Bm[None]) ** 2).sum(-1), 0)); sig = np.median(D[D > 0])
        Wd = np.exp(-(D / sig) ** 2); np.fill_diagonal(Wd, 0)
        res.setdefault("oracle_bouts", []).append((key, *eval_order(fiedler(Wd, eps=0, nvec=1)[0], mid), tt.max() + 1))
print("done %.0fs" % (time.time() - t0))
print("\n%-16s %8s %8s %8s %8s %8s  n>=0.7" % ("method", "med|rho|", "min", "max", "err%", "err_s"))
for nm, L in res.items():
    A = np.array([row[1:] for row in L], float)
    print("%-16s %8.3f %8.3f %8.3f %8.1f %8.0f  %d/%d" % (nm, np.median(np.abs(A[:, 0])), np.abs(A[:, 0]).min(),
                                                      np.abs(A[:, 0]).max(), 100 * np.median(A[:, 1]),
                                                      np.median(A[:, 2]), (np.abs(A[:, 0]) >= 0.7).sum(), len(A)))
print("\nper part (a_spec_k10 | b_dev200 | c_links | oracle_bouts) rho/err_s:")
for i, (r, h, ii, pur) in enumerate(parts):
    row = []
    for nm in ("a_spec_k10", "b_dev200_spec", "c_links1+knn10", "oracle_bouts"):
        m_ = [x for x in res.get(nm, []) if x[0] == (r, h)]
        row.append("%6.3f/%4.0f" % (m_[0][1], m_[0][3]) if m_ else "   -   ")
    print(r, h, len(ii), "  ".join(row))
