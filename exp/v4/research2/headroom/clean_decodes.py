"""Decode-level oracle results re-scored without the unlabeled-tail tiles (subjects 10 and 2), from the saved labels."""
import glob
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
keep = ~tail
f = lambda p: (macro_f1(y, p), macro_f1(y[keep], p[keep]))
z = np.load(os.path.join(HERE, "k7_base.npz")); b = f(z["fin"])
rb = f(D["F"]["ref_oof"].astype(np.int64))
log(f"{'k7_base':24s} fin all {b[0]:.4f} clean {b[1]:.4f} | refined (kernel ref_oof, = our refiner) all {rb[0]:.4f} clean {rb[1]:.4f}")
fu = f(np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64))
log(f"{'b4wa fused refined':24s} all {fu[0]:.4f} clean {fu[1]:.4f}")
for p in sorted(glob.glob(os.path.join(HERE, "k7_*.npz"))):
    nm = os.path.basename(p)[:-4]
    if nm == "k7_base":
        continue
    z = np.load(p); r = f(z["fin"]); msg = f"{nm:24s} fin all {r[0]:.4f} ({r[0] - b[0]:+.4f}) clean {r[1]:.4f} ({r[1] - b[1]:+.4f})"
    if "ref" in z.files:
        q = f(z["ref"]); msg += f" | refined all {q[0]:.4f} clean {q[1]:.4f}"
    log(msg)
