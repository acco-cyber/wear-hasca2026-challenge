"""extension 2 (tile level): same-limb chain neighbours that are our tiles carry the fit's OOF-decoded kernel output
QB (calibrated final probabilities, label-free for the tile's own subject: count prior nested by subject fold);
ctxQ = conf/decay-weighted SUM of log QB of those neighbours.  Also ctxB variants.  -> cache/ctxQ_oof_<fit>.npz"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
from f3b_ctxB import ctxB

T0 = time.time()
for fit, d in (("K7", K7), ("K9", K9)):
    st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True)
    y = st["oof_y"].astype(np.int64); sbj = st["oof_sbj"].astype(np.int64); fold = st["oof_fold"].astype(np.int64); sens = st["sensor_oof"].astype(np.int64)
    B2 = st["B2_OOF"].astype(np.float64); Q = np.log(np.clip(st["QB_OOF"].astype(np.float64), 1e-6, None)); Q = lsm(Q)
    print(f"{fit}: B2 {macro_f1(y, B2.argmax(1)):.4f} QB {macro_f1(y, Q.argmax(1)):.4f}")
    out = {}
    for Wn in (4, 8, 16):
        acc = np.zeros((len(y), N_CLS)); wsum = np.zeros(len(y))
        for s in np.unique(sbj):
            z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); rows = z["rows"]
            for L in range(4):
                mark = sens[rows] == L; p = np.flatnonzero(mark)
                a_, w_ = ctxB(Q[rows], mark, z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), p, Wn)
                acc[rows[p]] = a_; wsum[rows[p]] = w_
        out[f"W{Wn}"] = acc.astype(np.float32); out[f"n{Wn}"] = wsum.astype(np.float32)
        mean = acc / np.maximum(wsum, 1e-9)[:, None]
        print(f"  ctxQ W={Wn}: neighbour-mean argmax F1 (tiles with neighbours) {macro_f1(y[wsum > 0], mean.argmax(1)[wsum > 0]):.4f} |"
              + "".join(f" v{v} {macro_f1(y, lsm(B2 + v * acc).argmax(1)):.4f}" for v in (0.1, 0.2, 0.3, 0.5, 0.8)), flush=True)
        for v in (0.2, 0.3):
            pr = lsm(B2 + v * acc).argmax(1); print(f"     v{v} per fold " + " ".join(f"{macro_f1(y[fold == f], pr[fold == f]):.4f}/{macro_f1(y[fold == f], B2.argmax(1)[fold == f]):.4f}" for f in range(5)))
    np.savez(os.path.join(CACHE, f"ctxQ_oof_{fit}.npz"), **out)
print(f"[{time.time() - T0:.0f}s]")
