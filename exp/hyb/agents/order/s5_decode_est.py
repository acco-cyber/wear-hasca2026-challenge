"""Decoder along the ACTUAL best seriation estimate (c_links1+knn10 Fiedler, per full rec and per block-part), with the
orientation chosen using the true t (optimistic), plus noise-oracle sigma=45 for the precision threshold."""
import sys
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from o_common import *
from scipy.ndimage import uniform_filter1d
import hanbat_stack as hs

d = load_meta(); rec, t, y, sbj, fold = d["rec"], d["t"], d["y"], d["sbj"], d["fold"]
E = l2n(load_emb("oof"))
succ = np.load(K + r"\links_L0.npz")["oof_succ"]
P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy"); LP = np.log(np.clip(P, 1e-9, None))
dd = dict(sbj=sbj, sets={0: 2, 14: 2}); base = hs.macro_f1(y, hs.finish(P, dd))
inB1 = np.isin(y, B1); inB2 = np.isin(y, B2)


def seriate(ii):
    X = E[ii] - E[ii].mean(0)
    W, _ = knn_graph(X, k=10)
    loc = -np.ones(len(E), int); loc[ii] = np.arange(len(ii))
    sv = succ[ii]; ok = (sv >= 0) & (loc[np.maximum(sv, 0)] >= 0)
    a_ = np.flatnonzero(ok); b_ = loc[sv[ok]]
    Wl = sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(len(ii), len(ii))); Wl = Wl.maximum(Wl.T)
    f = fiedler_sp(W + Wl)[0]
    if spearmanr(f, t[ii]).correlation < 0:
        f = -f
    return rankdata(f) / len(f)


pos_full = np.zeros(len(t)); pos_part = np.zeros(len(t))
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); ii = ii[np.argsort(t[ii])]
    pos_full[ii] = seriate(ii)
    a = np.cumsum(inB1[ii]); b = np.cumsum(inB2[ii])
    s1 = a + (b[-1] - b); s2 = b + (a[-1] - a)
    k = s1.argmax() if s1.max() >= s2.max() else s2.argmax()
    for h, jj in enumerate((ii[:k + 1], ii[k + 1:])):
        pos_part[jj] = h + seriate(jj)          # part 0 first, then part 1 (block order given)


def decode(pos, w, beta):
    out = LP.copy()
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r); o = ii[np.argsort(pos[ii], kind="stable")]
        out[o] = (1 - beta) * LP[o] + beta * uniform_filter1d(LP[o], size=w, axis=0, mode="nearest")
    pr = hs.finish(np.exp(hs.lsm(out)), dd)
    return hs.macro_f1(y, pr), [round(hs.macro_f1(y[fold == f], pr[fold == f]), 4) for f in range(5)]


rng = np.random.default_rng(1)
for nm, pos in [("seriation_full", pos_full), ("seriation_blockparts", pos_part),
                ("noise45", t + rng.normal(0, 45, len(t)))]:
    best = max((decode(pos, w, b_) + (w, b_) for w in (3, 9, 31, 101) for b_ in (0.1, 0.3, 0.6)), key=lambda x: x[0])
    print("%-22s best F1 %.4f (d %+.4f) w=%d beta=%.1f per-fold %s" % (nm, best[0], best[0] - base, best[2], best[3],
                                                                    best[1]), flush=True)
