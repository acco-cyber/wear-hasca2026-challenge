"""Re-score the post-hoc oracles WITHOUT the unlabeled-tail tiles of subjects 10 (from second 2566) and 2 (from 3452)
(sessions recorded without labels -> 'null' in the OOF truth although the subject exercises; the kernel's
UNLABELLED_TAIL). Those tiles inflate the null->activity errors and the session oracle; the test has no such tails as far
as we know. Also the oracles restricted to test-like subjects."""
import numpy as np
from hlib import *

D = setup(); y, sbj = D["y"], D["sbj"]; n = len(y)
tail = np.zeros(n, bool)
for ii in D["order"]:
    s = sbj[ii[0]]
    if s == 10:
        tail[ii[2566:]] = True
    if s == 2:
        tail[ii[3452:]] = True
log(f"tail tiles {tail.sum()} (true null {np.mean(y[tail] == 0):.3f})")
keep = ~tail
M = np.load(os.path.join(HERE, "masks.npz")); MS, MN, MN3 = (M[k].astype(np.float64) for k in ("MS", "MN", "MN3"))
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64); lab = np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64)
f = lambda p, m=keep: macro_f1(y[m], p[m])
log(f"b4wa refined: all tiles {macro_f1(y, lab):.4f}; without tails {f(lab):.4f}; tail tiles decoded as activity {np.mean(lab[tail] > 0):.3f}")
bid, dist = bouts(D)
for nm, Mk in (("session(2-half)", MS), ("null(all)", MN), ("null<=3", MN3)):
    Qm = Qo * Mk; la = Qm.argmax(1); fx = lab.copy(); bad = Mk[np.arange(n), lab] == 0; fx[bad] = la[bad]
    log(f"  {nm:16s} oracle: all {macro_f1(y, lab):.4f}->{macro_f1(y, fx):.4f} ({macro_f1(y, fx) - macro_f1(y, lab):+.4f}) | "
        f"without tails {f(lab):.4f}->{f(fx):.4f} ({f(fx) - f(lab):+.4f}) [violations outside tails {np.sum(bad & keep)}]")
for d in (1, 2, 3):
    m = (lab != y) & (dist <= d); fx = lab.copy(); fx[m] = y[m]
    log(f"  all errors within {d} of a true edge fixed: without tails {f(lab):.4f}->{f(fx):.4f} ({f(fx) - f(lab):+.4f})")
err = lab != y
log(f"errors: all {err.sum()}, outside tails {np.sum(err & keep)}; null->act outside tails {np.sum(keep & (y == 0) & (lab > 0))}, "
    f"act->null {np.sum(keep & (y > 0) & (lab == 0))}, act->act' {np.sum(keep & (y > 0) & (lab > 0) & (lab != y))}")
for nm, fx_m in (("fix null->act", (y == 0) & (lab > 0)), ("fix act->null", (y > 0) & (lab == 0)), ("fix act->act'", (y > 0) & (lab > 0) & (lab != y))):
    fx = lab.copy(); fx[fx_m] = y[fx_m]; log(f"  {nm}: without tails {f(lab):.4f}->{f(fx):.4f} ({f(fx) - f(lab):+.4f})")
