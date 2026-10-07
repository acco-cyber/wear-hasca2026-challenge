"""step 2+3 (tile level): chain-context IMU expert on the OOF tiles (nested sim chains chain_lgb_s{s}.npz, nested IMU
log-probs), tile F1 vs the single-tile IMU and B2; blends B2' = lsm(B2 + w ctx [+ v ctxB]) with a nested (inner-fold) choice.
ctxB = conf/decay-weighted SUM of the B2 log-probs of same-limb chain neighbours that are our tiles (f3b_ctxB.py).
-> cache/ctx_oof.npz"""
import os, sys, time, itertools
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *

T0 = time.time()
st7 = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); st9 = np.load(os.path.join(K9, "stage.npz"), allow_pickle=True)
y = st7["oof_y"].astype(np.int64); sbj = st7["oof_sbj"].astype(np.int64); fold = st7["oof_fold"].astype(np.int64); sens = st7["sensor_oof"].astype(np.int64)
assert (st9["oof_y"] == y).all() and (st9["sensor_oof"] == sens).all()
B2 = {"K7": st7["B2_OOF"].astype(np.float64), "K9": st9["B2_OOF"].astype(np.float64)}
subs = [int(s) for s in np.unique(sbj)]
imu_tile = np.load(os.path.join(CACHE, "imu_oof_tile.npy"))
VARS = [(0, 0.9, "logp")] + [(Wn, dc, "logp") for Wn in (2, 4, 8, 16, 32) for dc in (0.9, 0.97)] + [(Wn, 0.9, "prob") for Wn in (4, 8, 16)]
CTX = {}
for Wn, dc, mode in VARS:
    out = np.zeros((len(y), N_CLS), np.float32); ws = np.zeros(len(y), np.float32)
    for s in subs:
        z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); rows = z["rows"]; A = np.load(os.path.join(CACHE, f"imu_oof_s{s}.npy"))
        for L in range(4):
            p = np.flatnonzero(sens[rows] == L)
            c, wsum = context(A[L], z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), p, Wn, decay=dc, mode=mode)
            out[rows[p]] = c; ws[rows[p]] = wsum
    if Wn == 0:
        assert np.abs(out - imu_tile).max() < 1e-4
    key_ = f"W{Wn}_d{dc}" + ("" if mode == "logp" else "_prob")
    CTX[key_] = out; pr = out.argmax(1)
    print(f"ctx {key_:14s}: tile F1 {macro_f1(y, pr):.4f} | per limb " + " ".join(f"L{L} {macro_f1(y[sens == L], pr[sens == L]):.4f}" for L in range(4))
          + f" | per fold " + " ".join(f"{macro_f1(y[fold == f], pr[fold == f]):.4f}" for f in range(5)) + f" | mean weight {ws.mean():.2f} [{time.time() - T0:.0f}s]", flush=True)
np.savez(os.path.join(CACHE, "ctx_oof.npz"), **CTX)
for k, b in B2.items():
    print(f"B2 {k} tile F1 {macro_f1(y, b.argmax(1)):.4f} | per fold " + " ".join(f"{macro_f1(y[fold == f], b.argmax(1)[fold == f]):.4f}" for f in range(5)))
WGT = [0.0, 0.1, 0.25, 0.5, 1.0]; VB = [0.0, 0.1, 0.2, 0.3, 0.5]


def nested(res, label, b):
    tab = {kk: [macro_f1(y[fold != f], pr[fold != f]) for f in range(5)] for kk, pr in res.items()}
    nest = np.zeros(len(y), np.int64); ch = []
    for f in range(5):
        best = max(tab, key=lambda kk: tab[kk][f]); ch.append(best); nest[fold == f] = res[best][fold == f]
    base = b.argmax(1)
    print(f"  NESTED {label}: choices {ch} -> tile F1 {macro_f1(y, nest):.4f} vs B2 {macro_f1(y, base):.4f} | per fold "
          + " ".join(f"{macro_f1(y[fold == f], nest[fold == f]):.4f}/{macro_f1(y[fold == f], base[fold == f]):.4f}" for f in range(5)), flush=True)
    return ch


for k, b in B2.items():
    print(f"--- {k}: B2' = lsm(B2 + w ctx) tile F1 overall | per fold")
    res = {}
    for key_, c in CTX.items():
        for w in WGT:
            if w == 0 and key_ != "W0_d0.9":
                continue
            pr = lsm(b + w * c.astype(np.float64)).argmax(1); res[(key_, w)] = pr
            if w in (0.25, 0.5, 1.0) and ("_d0.9" in key_):
                print(f"  {key_:14s} w={w}: {macro_f1(y, pr):.4f} | " + " ".join(f"{macro_f1(y[fold == f], pr[fold == f]):.4f}" for f in range(5)))
    nested(res, "IMU ctx (W, decay, w)", b)
    cb = np.load(os.path.join(CACHE, f"ctxB_oof_{k}.npz"))
    res2 = {}
    for WB in (4, 8, 16):
        for v in VB:
            for key_ in ("W0_d0.9", "W4_d0.9", "W8_d0.9", "W8_d0.9_prob"):
                for w in (0.0, 0.1, 0.25):
                    if (v == 0 and WB != 4):
                        continue
                    res2[(WB, v, key_, w)] = lsm(b + v * cb[f"W{WB}"].astype(np.float64) + w * CTX[key_].astype(np.float64)).argmax(1)
    for kk in [(4, 0.3, "W0_d0.9", 0.0), (8, 0.3, "W0_d0.9", 0.0), (8, 0.2, "W0_d0.9", 0.0), (8, 0.3, "W8_d0.9", 0.1)]:
        pr = res2[kk]; print(f"  ctxB {kk}: {macro_f1(y, pr):.4f} | " + " ".join(f"{macro_f1(y[fold == f], pr[fold == f]):.4f}" for f in range(5)))
    nested(res2, "ctxB + IMU ctx (WB, v, ctx, w)", b)
print(f"done [{time.time() - T0:.0f}s]")
