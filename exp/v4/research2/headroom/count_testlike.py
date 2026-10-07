"""True-count oracle restricted to test-like subjects (fused b4wa, finish-only re-Sinkhorn): the test subjects perform
every exercise exactly once (participant_meta_data: 9+8+1 / 9+9 activities), unlike training subjects 0 and 14 (two
sets) and 2 (17 exercises)."""
import numpy as np
from hlib import *

D = setup(); y, sbj = D["y"], D["sbj"]
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64)
tg_f = {int(s): Qo[sbj == s].sum(0) for s in np.unique(sbj)}; TT = true_counts(D)
base = macro_f1(y, calibrate_targets(Qo, sbj, tg_f).argmax(1))


def with_true(subs):
    tg = {s: (TT[s] if s in subs else tg_f[s]) for s in tg_f}
    return macro_f1(y, calibrate_targets(Qo, sbj, tg).argmax(1))


allS = set(tg_f)
log(f"b4wa finish learned {base:.4f}; true all {with_true(allS):.4f}; true except 0,2,14 {with_true(allS - {0, 2, 14}):.4f}; "
    f"true except 0,2,14,10 {with_true(allS - {0, 2, 14, 10}):.4f}; true only 0,2,14 {with_true({0, 2, 14}):.4f}")
log("one subject at a time (overall F1 gain): " + " ".join(f"{s}:{with_true({s}) - base:+.4f}" for s in sorted(allS)))
# exercise split only, test-like subjects
tg = {s: (np.r_[tg_f[s][0], TT[s][1:] / TT[s][1:].sum() * tg_f[s][1:].sum()] if s not in (0, 2, 14) else tg_f[s]) for s in tg_f}
log(f"true exercise split (learned null) on test-like subjects only: {macro_f1(y, calibrate_targets(Qo, sbj, tg).argmax(1)):.4f}")
# the same through K7's refiner? (finish-only, pre-refiner numbers are what we compare)
