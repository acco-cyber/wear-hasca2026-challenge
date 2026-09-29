"""OOF macro-F1 grid: method x a x mask(all rows | finish-predicted activity rows) x null_mode.
python grid2.py <methods comma> <a list> <masks comma> <null_modes comma>
methods may be 'oracleQ' (true half with confidence Q, ceiling only)."""
import os, sys
import numpy as np
from common import *
sm, P = load_oof(); y, fold, sbj = sm["y"], sm["fold"], sm["sbj"]
f0, pf0, pred0 = score(y, fold, sbj, P)
print(f"baseline {f0:.4f} per-fold {np.round(pf0, 4).tolist()}", flush=True)
cache = dict(np.load("oof_p1.npz"))
th = true_half(sm)
names = sys.argv[1].split(","); A = [float(x) for x in sys.argv[2].split(",")]
masks = sys.argv[3].split(",") if len(sys.argv) > 3 else ["all"]
modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["keepnull"]
MASK = {"all": None, "predact": pred0 > 0, "argmaxact": P.argmax(1) > 0}
for nm in names:
    p1 = np.where(th == 1, float(nm[6:]), 1 - float(nm[6:])) if nm.startswith("oracle") else cache[nm]
    for mk in masks:
        for md in modes:
            for a in A:
                f, pf, _ = score(y, fold, sbj, apply_prior(P, p1, a, md, MASK[mk]))
                d = np.array(pf) - pf0
                print(f"{nm:18s} {mk:8s} {md:8s} a={a:<4} F1 {f:.4f} ({f-f0:+.4f})  per-fold d {' '.join(f'{x:+.4f}' for x in d)}"
                      f"  folds>0 {(d > 0).sum()}/5", flush=True)
