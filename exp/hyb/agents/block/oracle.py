"""Ceiling: apply the TRUE half-session block as the prior (not a method, an upper bound)."""
from common import *
sm, P = load_oof(); y, fold, sbj = sm["y"], sm["fold"], sm["sbj"]
f0, pf0, _ = score(y, fold, sbj, P); print(f"baseline {f0:.4f}", np.round(pf0, 4))
th = true_half(sm); a_ = y > 0
print("true-half agrees with true block on activity rows", np.mean(th[a_] == BLK[y[a_]]))
for conf in (0.99, 0.9, 0.8, 0.7):
    p1 = np.where(th == 1, conf, 1 - conf)
    for a in (0.5, 1, 2):
        for nm in ("none", "geo"):
            f, pf, _ = score(y, fold, sbj, apply_prior(P, p1, a, nm))
            print(f"oracle conf {conf} a {a} null {nm}: {f:.4f} ({f-f0:+.4f})", np.round(np.array(pf) - pf0, 4), flush=True)
# oracle using activity-row true block only (null rows untouched)
p1 = np.where(a_, np.where(BLK[y] == 1, 0.99, 0.01), 0.5)
f, pf, _ = score(y, fold, sbj, apply_prior(P, p1, 1)); print(f"oracle activity-only 0.99 a1: {f:.4f} ({f-f0:+.4f})")
