"""Lever 1: count targets. K7 single fit, OOF only.
 base      : learned nested counts (reproduces v4_local: pre-refiner 0.9290, refined 0.9311)
 finish    : graph decode with the learned counts, ORACLE true counts only in the final Sinkhorn
 true      : ORACLE true per-(subject,class) counts in the pass-1 decode AND the final Sinkhorn
 mix0.5/0.75: count error shrunk to 50% / 25% of the learned error (oracle-assisted, interpolation)"""
import numpy as np
from hlib import *

D = setup(); y, fold, sbj = D["y"], D["fold"], D["sbj"]
log(f"K7 kernel QB {macro_f1(y, D['F']['QB_OOF'].argmax(1)):.4f} ref {macro_f1(y, D['F']['ref_oof']):.4f}")
res = {}
r = decode(D, D["Lo"], "learned"); P0 = r["P0"]
res["base"] = (macro_f1(y, r["fin"]), macro_f1(y, refine_oof(D, r["fin"], D["Lo"], r["P"], r["Q"])))
np.savez_compressed(os.path.join(HERE, "k7_base.npz"), P0=P0.astype(np.float32), P=r["P"].astype(np.float32), Q=r["Q"].astype(np.float32), fin=r["fin"])
Qf = finish_targets(r["P"], sbj, true_counts(D)); ff = Qf.argmax(1)
log(f"finish-only true counts: pre-refiner {macro_f1(y, ff):.4f}")
res["finish_true"] = (macro_f1(y, ff), macro_f1(y, refine_oof(D, ff, D["Lo"], r["P"], Qf)))
for nm, cm, lam in (("true", "true", 1.0), ("mix0.5", "mix", 0.5), ("mix0.75", "mix", 0.75)):
    r2 = decode(D, D["Lo"], cm, lam, P0=P0)
    res[nm] = (macro_f1(y, r2["fin"]), macro_f1(y, refine_oof(D, r2["fin"], D["Lo"], r2["P"], r2["Q"])))
    if nm == "true":
        np.savez_compressed(os.path.join(HERE, "k7_truecnt.npz"), P=r2["P"].astype(np.float32), Q=r2["Q"].astype(np.float32), fin=r2["fin"])
for k, v in res.items():
    log(f"RESULT counts {k:12s} pre-refiner {v[0]:.4f} refined {v[1]:.4f}")
