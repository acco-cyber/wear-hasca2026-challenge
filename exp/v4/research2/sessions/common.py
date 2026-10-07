"""shared loaders for the session-structure study (read-only inputs)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"
sys.path.insert(0, os.path.join(W, "exp", "hyb"))
from hanbat_stack import macro_f1  # noqa
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
OUT = os.path.join(W, "exp", "v4", "research2", "sessions")
A = np.array([1, 2, 3, 6, 7, 11, 12, 13, 14])
B = np.array([4, 5, 8, 9, 10, 15, 16, 17, 18])
BLK = np.zeros(19, int); BLK[A] = 1; BLK[B] = 2           # 0 = null, 1 = block A, 2 = block B
SESS_TEST = {22: [2430, 2410, 177], 23: [1440, 1581, 107], 24: [1045, 950], 25: [965, 949]}
NACT_TEST = {22: [9, 8, 1], 23: [9, 8, 1], 24: [9, 9], 25: [9, 9]}


def base():
    st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
    d = {k: st[k] for k in ["oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "test_sbj", "ref_oof", "ref_test", "QB_OOF", "QB_TEST",
                            "true_succ", "sensor_oof", "sensor_test"]}
    d["t"] = d["oof_start"] // 50
    return d


def true_block(d):
    """per OOF tile: block (1/2) of the time-nearest non-null tile of its recording"""
    y, rec, t = d["oof_y"], d["oof_rec"], d["t"]
    out = np.zeros(len(y), int)
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r); o = ii[np.argsort(t[ii])]
        b = BLK[y[o]]; nz = np.flatnonzero(b > 0); pos = np.arange(len(o))
        k = np.searchsorted(nz, pos).clip(1, len(nz) - 1)
        left, right = nz[k - 1], nz[k]
        near = np.where(np.abs(pos - left) <= np.abs(right - pos), left, right)
        out[o] = b[near]
    return out


def ordered(d, r):
    ii = np.flatnonzero(d["oof_rec"] == r); return ii[np.argsort(d["t"][ii])]


def f1(y, p):
    return macro_f1(y, p)
