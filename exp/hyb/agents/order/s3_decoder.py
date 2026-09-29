"""Decoder along an (estimated) time order: sort windows of each recording by position estimate, smooth class log-probs
with a centred moving average of w windows, mix with original (beta), then the standard finish() -> macro-F1.
Positions tested: true t (oracle), true t + Gaussian noise sigma (to find the precision needed), and the best seriation
estimates from s1/s2 (b_dev200 spectral, SFA spectral) mapped to rank-positions."""
import sys
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from o_common import *
from scipy.ndimage import uniform_filter1d
import hanbat_stack as hs

d = load_meta(); rec, t, y, sbj, fold = d["rec"], d["t"], d["y"], d["sbj"], d["fold"]
P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy")
LP = np.log(np.clip(P, 1e-9, None))
dd = dict(sbj=sbj, sets={0: 2, 14: 2})
base = hs.macro_f1(y, hs.finish(P, dd))
rng = np.random.default_rng(0)
recs = np.unique(rec)


def decode(pos, w, beta):
    out = LP.copy()
    for r in recs:
        ii = np.flatnonzero(rec == r)
        o = ii[np.argsort(pos[ii], kind="stable")]
        sm = uniform_filter1d(LP[o], size=w, axis=0, mode="nearest")
        out[o] = (1 - beta) * LP[o] + beta * sm
    Q = np.exp(hs.lsm(out))
    pr = hs.finish(Q, dd)
    return hs.macro_f1(y, pr), [hs.macro_f1(y[fold == f], pr[fold == f]) for f in range(5)]


print("baseline %.4f" % base)
for name, sig in [("oracle", 0), ("noise10", 10), ("noise30", 30), ("noise60", 60), ("noise120", 120),
                  ("noise300", 300), ("noise550", 550)]:
    pos = t + (rng.normal(0, sig, len(t)) if sig else 0)
    best = None
    for w in (3, 5, 9, 15, 31):
        for beta in (0.3, 0.6, 0.9):
            f1, pf = decode(pos, w, beta)
            if best is None or f1 > best[0]:
                best = (f1, w, beta, pf)
    print("%-10s best F1 %.4f (d %+.4f) w=%d beta=%.1f  per-fold %s" % (name, best[0], best[0] - base, best[1], best[2],
                                                                     [round(x, 4) for x in best[3]]), flush=True)
