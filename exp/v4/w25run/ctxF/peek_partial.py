"""early look on the folds already finished (IMU F1, ctx F1, blends) - informational only"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
y = st["oof_y"].astype(np.int64); sbj = st["oof_sbj"].astype(np.int64); fold = st["oof_fold"].astype(np.int64); sens = st["sensor_oof"].astype(np.int64)
B2 = st["B2_OOF"].astype(np.float64)
fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
done = [f for f in range(5) if os.path.exists(os.path.join(CACHE, f"imu_raw_f{f}.npy"))]
subs = [int(s) for s in np.unique(sbj)]
# rebuild the key order of f2_imu
Z = {s: np.load(os.path.join(CACHE, f"imuf_s{s}.npz"))["rows"] for s in subs}
fo = np.concatenate([np.full(len(Z[s]), fold_of[s]) for s in subs for L in range(4)])
keyS = np.concatenate([np.full(len(Z[s]), s) for s in subs for L in range(4)]); keyL = np.concatenate([np.full(len(Z[s]), L) for s in subs for L in range(4)])
keyP = np.concatenate([np.arange(len(Z[s])) for s in subs for L in range(4)])
lp = np.full((len(fo), N_CLS), np.nan, np.float32)
for f in done:
    lp[fo == f] = lsm(np.load(os.path.join(CACHE, f"imu_raw_f{f}.npy")).astype(np.float64))
m_t = np.isin(fold, done)
for Wn in (0, 4, 8, 16, 32):
    out = np.zeros((len(y), N_CLS), np.float32)
    for s in subs:
        if fold_of[s] not in done:
            continue
        z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); rows = z["rows"]
        for L in range(4):
            A = lp[(keyS == s) & (keyL == L)]; assert (keyP[(keyS == s) & (keyL == L)] == np.arange(len(rows))).all()
            p = np.flatnonzero(sens[rows] == L)
            c, _ = context(A, z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), p, Wn)
            out[rows[p]] = c
    pr = out.argmax(1)
    line = f"W={Wn:2d}: ctx F1 {macro_f1(y[m_t], pr[m_t]):.4f} |"
    for w in (0.25, 0.5, 1.0):
        line += f" w{w} {macro_f1(y[m_t], lsm(B2 + w * out).argmax(1)[m_t]):.4f}"
    print(line + f" | B2 {macro_f1(y[m_t], B2.argmax(1)[m_t]):.4f} (folds {done})", flush=True)
