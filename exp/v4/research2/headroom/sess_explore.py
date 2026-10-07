"""Where are the physical session boundaries in the training recordings? bout sequence + largest video/gravity jumps"""
import numpy as np
from hlib import *
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; BLK = np.zeros(19, int); BLK[B1] = 1; BLK[[c for c in range(1, 19) if c not in B1]] = 2
D = setup(); y, sbj = D["y"], D["sbj"]
vm = D["F"]["sc_o"][3]; vm = vm / (np.linalg.norm(vm, axis=1, keepdims=True) + 1e-6)
for r_i, ii in enumerate(D["order"]):
    lab = y[ii]; chg = np.r_[True, lab[1:] != lab[:-1]]; st = np.flatnonzero(chg); ln = np.diff(np.r_[st, len(ii)])
    seq = [(int(lab[s]), int(l)) for s, l in zip(st, ln)]
    jump = np.r_[0, np.linalg.norm(vm[ii][1:] - vm[ii][:-1], axis=1)]
    med = np.median(jump); top = np.argsort(-jump)[:6]
    print(f"rec {D['rec'][ii[0]]} sbj {sbj[ii[0]]} n={len(ii)} start-gaps {np.unique(np.diff(D['start'][ii]))[:5]}")
    print("   bouts: " + " ".join(f"{c}{'ab'[BLK[c] - 1] if c else ''}:{l}" for c, l in seq))
    print("   top video jumps (pos, jump/median): " + " ".join(f"{p}:{jump[p] / med:.1f}" for p in sorted(top)))
