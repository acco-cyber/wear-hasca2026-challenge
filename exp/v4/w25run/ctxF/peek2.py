"""diagnostics on the finished folds: chain confidence, oracle-chain ceiling, conf_pow / prob variants"""
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
Z = {s: np.load(os.path.join(CACHE, f"imuf_s{s}.npz"))["rows"] for s in subs}
fo = np.concatenate([np.full(len(Z[s]), fold_of[s]) for s in subs for L in range(4)])
keyS = np.concatenate([np.full(len(Z[s]), s) for s in subs for L in range(4)]); keyL = np.concatenate([np.full(len(Z[s]), L) for s in subs for L in range(4)])
lp = np.full((len(fo), N_CLS), np.nan, np.float32)
for f in done:
    lp[fo == f] = lsm(np.load(os.path.join(CACHE, f"imu_raw_f{f}.npy")).astype(np.float64))
m_t = np.isin(fold, done)
confs = []; corr = []
for s in subs:
    z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz"))
    for L in range(4):
        su = z[f"succ{L}"]; ok = su >= 0; confs.append(z[f"conf{L}"][ok]); corr.append((su == z["true_loc"])[ok])
confs = np.concatenate(confs); corr = np.concatenate(corr)
print("chain conf quantiles", np.quantile(confs, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3), "| precision by conf bin:",
      [(lo, round(float(corr[(confs >= lo) & (confs < lo + 0.2)].mean()), 3)) for lo in (0, 0.2, 0.4, 0.6, 0.8)])


def run(kind, Wn, decay=0.9, conf_pow=1.0, mode="logp"):
    out = np.zeros((len(y), N_CLS), np.float32)
    for s in subs:
        if fold_of[s] not in done:
            continue
        z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); rows = z["rows"]
        for L in range(4):
            A = lp[(keyS == s) & (keyL == L)]; p = np.flatnonzero(sens[rows] == L)
            if kind == "oracle":
                su = z["true_loc"].astype(np.int64); cf = np.ones(len(su), np.float32)
            else:
                su = z[f"succ{L}"].astype(np.int64); cf = z[f"conf{L}"].astype(np.float32)
            c, _ = context(A, su, cf, p, Wn, decay=decay, conf_pow=conf_pow, mode=mode)
            out[rows[p]] = c
    pr = out.argmax(1)
    line = f"{kind:6s} W={Wn:2d} d={decay} cp={conf_pow} {mode}: ctx F1 {macro_f1(y[m_t], pr[m_t]):.4f} |"
    for w in (0.1, 0.25, 0.5):
        line += f" w{w} {macro_f1(y[m_t], lsm(B2 + w * out).argmax(1)[m_t]):.4f}"
    print(line, flush=True)


for Wn in (2, 8, 32):
    run("oracle", Wn, decay=1.0)
run("oracle", 8, decay=1.0, mode="prob")
run("lgb", 8, decay=1.0, conf_pow=0.0)
run("lgb", 8, decay=0.9, conf_pow=0.5)
run("lgb", 8, decay=0.9, mode="prob")
print("B2", macro_f1(y[m_t], B2.argmax(1)[m_t]))
