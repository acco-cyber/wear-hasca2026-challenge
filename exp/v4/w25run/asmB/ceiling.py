"""diagnostic (scoring-side): with ORACLE fragment alignment, how many time coordinates could become complete (all 4
limb nodes placed) when only fragments holding >= 1 of our anchored tiles can be aligned across limbs"""
import numpy as np
from asmlib import *
S = stage(); subs = [int(x) for x in np.unique(S["oof_sbj"])]
for thr in (0.5, 0.7, 0.9):
    tot = 0; comp = 0; comp3 = 0
    for s in subs:
        d = load_sim(s); n = d["n"]; tr = d["truth"]
        frag, off, flim, flen, fn = fragments(d["succ"], d["conf"], thr)
        hasown = np.zeros(len(flim), bool)
        for L in range(4):
            q = np.flatnonzero(d["owner"][L] >= 0); hasown[frag[L, q]] = True
        al = np.zeros((4, n), bool)                   # by true position
        for L in range(4):
            al[L, tr["tnode"][L]] = hasown[frag[L]]
        k = al.sum(0); tot += n; comp += (k == 4).sum(); comp3 += (k >= 3).sum()
    print(f"thr {thr}: coordinates whose 4 limb nodes all sit in fragments holding an own tile {comp/tot:.3f} (>=3 limbs {comp3/tot:.3f})", flush=True)
