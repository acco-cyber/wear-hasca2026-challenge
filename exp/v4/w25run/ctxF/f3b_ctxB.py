"""extension: same-limb chain neighbours that are OUR tiles (same limb, so their 2025 twin lies on the chain) carry the
stage-B base B2 of that tile; weighted sum of their B2 log-probs (weights cum. conf x decay^k), OOF on the nested sim chains.
-> cache/ctxB_oof_<fit>.npz (W{W}: weighted-sum log-probs, n{W}: weight sums)"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *


def ctxB(B2rows, mark, su, cf, start, Wn, decay=0.9):
    pr, cb = preds_arrays(su, cf)
    Ff, Cf = walks(su, cf, start, Wn); Fb, Cb = walks(pr, cb, start, Wn)
    acc = np.zeros((len(start), N_CLS)); wsum = np.zeros(len(start))
    for F, C in ((Ff, Cf), (Fb, Cb)):
        for k in range(Wn):
            ok = (F[:, k] >= 0) & mark[np.maximum(F[:, k], 0)]
            w = C[:, k].astype(np.float64) * decay ** (k + 1) * ok
            acc += w[:, None] * B2rows[np.maximum(F[:, k], 0)]; wsum += w
    return acc, wsum


if __name__ == "__main__":
    T0 = time.time()
    for fit, d in (("K7", K7), ("K9", K9)):
        st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True)
        y = st["oof_y"].astype(np.int64); sbj = st["oof_sbj"].astype(np.int64); fold = st["oof_fold"].astype(np.int64); sens = st["sensor_oof"].astype(np.int64)
        B2 = st["B2_OOF"].astype(np.float64); out = {}
        for Wn in (4, 8, 16, 32):
            acc = np.zeros((len(y), N_CLS)); wsum = np.zeros(len(y))
            for s in np.unique(sbj):
                z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); rows = z["rows"]
                for L in range(4):
                    mark = sens[rows] == L; p = np.flatnonzero(mark)
                    a_, w_ = ctxB(B2[rows], mark, z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), p, Wn)
                    acc[rows[p]] = a_; wsum[rows[p]] = w_
            out[f"W{Wn}"] = acc.astype(np.float32); out[f"n{Wn}"] = wsum.astype(np.float32)
            line = f"{fit} ctxB W={Wn}: mean neighbour weight {wsum.mean():.2f}, share with any {np.mean(wsum > 0):.3f} |"
            for v in (0.1, 0.2, 0.3, 0.5, 0.8):
                pr = lsm(B2 + v * acc).argmax(1); line += f" v{v} {macro_f1(y, pr):.4f}"
            print(line + f" | B2 {macro_f1(y, B2.argmax(1)):.4f} [{time.time() - T0:.0f}s]", flush=True)
        np.savez(os.path.join(CACHE, f"ctxB_oof_{fit}.npz"), **out)
