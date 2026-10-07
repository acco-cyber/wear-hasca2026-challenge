"""Fused b4wa finish-only: count error shrunk on the test-like subjects only (not 0, 2, 14), and the gain split by
count-error size (which subject-class keys carry the gain)."""
import numpy as np
from hlib import *

D = setup(); y, sbj = D["y"], D["sbj"]
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64)
tg_f = {int(s): Qo[sbj == s].sum(0) for s in np.unique(sbj)}; TT = true_counts(D)
f = lambda tg: macro_f1(y, calibrate_targets(Qo, sbj, tg).argmax(1))
base = f(tg_f); TL = [s for s in tg_f if s not in (0, 2, 14)]
for lam in (0.25, 0.5, 0.75, 1.0):
    tg = {s: (tg_f[s] + lam * (TT[s] - tg_f[s]) if s in TL else tg_f[s]) for s in tg_f}
    log(f"test-like subjects, count error x{1 - lam:.2f}: {f(tg):.4f} ({f(tg) - base:+.4f})")
# only keys with |error| > k get the true count (are the gains in a few big misses?)
err = {s: TT[s] - tg_f[s] for s in tg_f}
for k in (5, 10, 15, 20):
    tg = {}
    for s in tg_f:
        t = np.where(np.abs(err[s]) > k, TT[s], tg_f[s]) if s in TL else tg_f[s].copy()
        t[0] = max(TT[s].sum() - t[1:].sum(), 0.05 * TT[s].sum()); tg[s] = t
    nk = sum(int(np.sum(np.abs(err[s][1:]) > k)) for s in TL)
    log(f"test-like, true count only where |error| > {k:2d} tiles ({nk} of {18 * len(TL)} exercise keys): {f(tg):.4f} ({f(tg) - base:+.4f})")
