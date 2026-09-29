"""Task 1: how cleanly do the true blocks B1/B2 split in time per recording?"""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]
blk = np.zeros(19, int); blk[B1] = 1; blk[B2] = 2
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
y, rec, st, sbj = sm["y"], sm["rec"], sm["start"], sm["sbj"]
tot_ok = tot = 0
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); ii = ii[np.argsort(st[ii])]
    yy = y[ii]; a = yy > 0; b = blk[yy[a]]
    n = len(b)
    # best single switch point: first part block X, second block Y (X != Y)
    c1 = np.r_[0, np.cumsum(b == 1)]; c2 = np.r_[0, np.cumsum(b == 2)]
    # ok(k) for B1-then-B2 = #B1 in [0,k) + #B2 in [k,n)
    ok12 = c1 + (c2[-1] - c2); ok21 = c2 + (c1[-1] - c1)
    k12, k21 = ok12.argmax(), ok21.argmax()
    if ok12[k12] >= ok21[k21]:
        order, k, ok = "B1->B2", k12, ok12[k12]
    else:
        order, k, ok = "B2->B1", k21, ok21[k21]
    tot_ok += ok; tot += n
    # bout sequence (>=10s runs) of block ids
    cut = np.flatnonzero(np.diff(yy) != 0) + 1; s_ = np.r_[0, cut]; e_ = np.r_[cut, len(yy)]
    seq = [(int(yy[p]), int(q - p)) for p, q in zip(s_, e_) if yy[p] > 0 and q - p >= 10]
    bseq = "".join("a" if blk[c] == 1 else "b" for c, _ in seq)
    # number of block switches in bout sequence
    nsw = sum(bseq[i] != bseq[i + 1] for i in range(len(bseq) - 1))
    print(f"rec {r:2d} sbj {sbj[ii[0]]:2d} n_act {n:5d} {order} switch@{k/n:.2f} purity {ok/n:.4f} "
          f"bout-switches {nsw:2d}  {bseq}")
print(f"overall purity {tot_ok/tot:.4f}")
