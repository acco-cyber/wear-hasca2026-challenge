"""Task 3: OOF macro-F1 of P' = block prior (p1^a on B1, (1-p1)^a on B2, null untouched) -> finish()."""
import os, sys
import numpy as np
from common import *
sm, P = load_oof(); y, fold, sbj = sm["y"], sm["fold"], sm["sbj"]
f0, pf0, _ = score(y, fold, sbj, P)
print(f"baseline {f0:.4f} per-fold {np.round(pf0, 4).tolist()}")
cache = dict(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else "oof_p1.npz")))
names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(cache)
A = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0.25", "0.5", "1", "2", "4"])]
for nm in names:
    p1 = cache[nm]
    for a in A:
        f, pf, _ = score(y, fold, sbj, apply_prior(P, p1, a))
        d = np.array(pf) - pf0
        print(f"{nm:18s} a={a:<5} F1 {f:.4f} ({f-f0:+.4f})  per-fold d {' '.join(f'{x:+.4f}' for x in d)}  "
              f"folds>0: {(d > 0).sum()}/5", flush=True)
