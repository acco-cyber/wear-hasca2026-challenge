"""step 1b: IMU-only LightGBM (19 classes) on all four limbs of every training tile position, nested by subject fold
(oof_fold); plus a full model (all 22 subjects) for test.  -> cache/imu_oof_s{s}.npy (4, n, 19) log-probs, cache/imu_full.txt"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
sys.path.insert(0, FEAS)
from common import load_stage

P = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.1, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.4,
         bagging_fraction=0.5, bagging_freq=1, lambda_l2=2.0, max_bin=63, verbose=-1, num_threads=2)
ROUNDS = 250
T0 = time.time()
S = load_stage(); subs = [int(x) for x in np.unique(S["oof_sbj"])]
fold_of = {int(s): int(f) for s, f in zip(S["oof_sbj"], S["oof_fold"])}
Z = {s: np.load(os.path.join(CACHE, f"imuf_s{s}.npz")) for s in subs}
X, y, fo, key = [], [], [], []
for s in subs:
    F = Z[s]["F"]; rows = Z[s]["rows"]; n = len(rows)
    for L in range(4):
        X.append(F[L]); y.append(S["oof_y"][rows]); fo.append(np.full(n, fold_of[s])); key.append(np.stack([np.full(n, s), np.full(n, L), np.arange(n)], 1))
X = np.concatenate(X); y = np.concatenate(y); fo = np.concatenate(fo); key = np.concatenate(key)
print(f"{len(y)} rows x {X.shape[1]} features [{time.time() - T0:.0f}s]", flush=True)


def fit(f):
    pf = os.path.join(CACHE, f"imu_raw_f{f}.npy"); pm = os.path.join(CACHE, "imu_full.txt")
    if f >= 0 and os.path.exists(pf):
        return f, np.load(pf)
    if f < 0 and os.path.exists(pm):
        return f, None
    tr = fo != f if f >= 0 else np.ones(len(y), bool)
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr]), ROUNDS)
    if f < 0:
        m.save_model(pm); return f, None
    te = fo == f; raw = m.predict(X[te], raw_score=True); np.save(pf, raw.astype(np.float32))
    print(f"fold {f} done [{time.time() - T0:.0f}s]", flush=True)
    return f, raw


out = Parallel(n_jobs=3)(delayed(fit)(f) for f in [0, 1, 2, 3, 4, -1])
lp = np.zeros((len(y), N_CLS), np.float32)
for f, raw in out:
    if raw is not None:
        lp[fo == f] = lsm(raw.astype(np.float64))
pred = lp.argmax(1)
print(f"[{time.time() - T0:.0f}s] nested IMU tile macro-F1 all limbs {macro_f1(y, pred):.4f} | per limb "
      + " ".join(f"L{L} {macro_f1(y[key[:, 1] == L], pred[key[:, 1] == L]):.4f}" for L in range(4))
      + " | per fold " + " ".join(f"{macro_f1(y[fo == f], pred[fo == f]):.4f}" for f in range(5)), flush=True)
# OOF tile view: the limb each OOF tile actually carries
sens = S["sensor_oof"]
for s in subs:
    m = key[:, 0] == s; n = len(Z[s]["rows"])
    A = np.zeros((4, n, N_CLS), np.float32)
    for L in range(4):
        mm = m & (key[:, 1] == L); A[L, key[mm, 2]] = lp[mm]
    np.save(os.path.join(CACHE, f"imu_oof_s{s}.npy"), A)
own = np.zeros((len(S["oof_y"]), N_CLS), np.float32)
for s in subs:
    rows = Z[s]["rows"]; A = np.load(os.path.join(CACHE, f"imu_oof_s{s}.npy")); own[rows] = A[sens[rows], np.arange(len(rows))]
np.save(os.path.join(CACHE, "imu_oof_tile.npy"), own)
yo = S["oof_y"]
print(f"IMU on the OOF tiles' own limb: macro-F1 {macro_f1(yo, own.argmax(1)):.4f} | per limb "
      + " ".join(f"L{L} {macro_f1(yo[sens == L], own[sens == L].argmax(1)):.4f}" for L in range(4)), flush=True)
