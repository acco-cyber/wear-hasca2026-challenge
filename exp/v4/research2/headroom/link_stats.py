"""Link quality of the oracle link sets: exact successor and cross-label (harmful) rate"""
import numpy as np
from hlib import *

D = setup(); y = D["y"]


def stats(nm, L):
    ex, cl, ln = [], [], []
    for su, _ in L:
        m = su >= 0; ln.append(m.mean()); ex.append((su[m] == D["ts"][m]).mean()); cl.append((y[su[m]] != y[m]).mean())
    log(f"{nm:14s}: linked {np.mean(ln):.3f} exact {np.mean(ex):.4f} cross-label {np.mean(cl):.4f} (null-involved {np.mean([(((y[su[su >= 0]] == 0) | (y[su >= 0] == 0)) & (y[su[su >= 0]] != y[su >= 0])).mean() for su, _ in L]):.4f})")


stats("kernel", D["Lo"]); stats("true", true_links(D)[:1])
for f in (0.5, 0.8):
    stats(f"corrected {f}", corrected_links(D, f))
