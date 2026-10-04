"""Second round of the public notebook's pseudo-label adaptation of the tabular expert T, on CPU:
T (hand-crafted per-tile features) is re-fitted with each new subject's own tiles labelled by the CURRENT best
pipeline output (cross-fitted in two halves), then the blend B2 = 0.2 window + 0.5 S3 + 0.3 adapted-T is rebuilt.
  python adapt_local.py <oof_pseudo_Q.npy> <test_pseudo_Q.npy> <tag> [--stage work/good/keep3]
  -> work/good/b2_<tag>_oof.npy, b2_<tag>_test.npy (log-probs), ta_<tag>_{oof,test}.npy"""
import os, sys, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import macro_f1, KEEP, N_CLS, lsm
W = r"E:\Claude code\wear"
TAB_PARAMS = dict(objective="multiclass", num_class=19, learning_rate=0.08, num_leaves=31, min_data_in_leaf=80, feature_fraction=0.3,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63, num_threads=8, verbose=-1, seed=0)
PS_PARAMS = dict(TAB_PARAMS, min_data_in_leaf=40); PS_ROUNDS, PS_WEIGHT = 150, 3.0


def X_of(split):
    f = np.load(os.path.join(KEEP, f"feat_{split}.npz")); return np.concatenate([f["imu"], f["vmot"], f["vpca"]], 1).astype(np.float32)


def adapt_T(X_lab, y_lab, X_new, pseudo, seed):
    half = np.random.default_rng(seed).integers(0, 2, len(X_new)); out = np.zeros((len(X_new), N_CLS), np.float32)
    for h in (0, 1):
        ps, te = half == h, half != h
        b = lgb.train(PS_PARAMS, lgb.Dataset(np.concatenate([X_lab, X_new[ps]]), np.concatenate([y_lab, pseudo[ps]]),
                                             weight=np.concatenate([np.ones(len(y_lab)), np.full(ps.sum(), PS_WEIGHT)]), params={"max_bin": 63}), PS_ROUNDS)
        out[te] = np.log(np.clip(b.predict(X_new[te]), 1e-7, 1))
    return out


ap = argparse.ArgumentParser(); ap.add_argument("oof_q"); ap.add_argument("test_q"); ap.add_argument("tag"); ap.add_argument("--stage", default=os.path.join(W, "work", "good", "keep3"))
a = ap.parse_args(); t0 = time.time()
st = np.load(os.path.join(a.stage, "stage.npz")); y, fold = st["oof_y"].astype(np.int64), st["oof_fold"].astype(np.int64)
XO, XT = X_of("oof"), X_of("test"); pl_o = np.load(a.oof_q).argmax(1); pl_t = np.load(a.test_q).argmax(1)
print(f"pseudo-label quality (OOF F1) {macro_f1(y, pl_o):.4f}; kernel's adapted T (tile-level) {macro_f1(y, st['TA_OOF'].argmax(1)):.4f}", flush=True)
TA = np.zeros((len(y), N_CLS), np.float32)
for k in range(5):
    tr, va = fold != k, fold == k; TA[va] = adapt_T(XO[tr], y[tr], XO[va], pl_o[va], k); print(f"fold {k} done [{time.time() - t0:.0f}s]", flush=True)
print(f"second-round adapted T (tile-level OOF F1) {macro_f1(y, TA.argmax(1)):.4f}", flush=True)
TT = adapt_T(XO, y, XT, pl_t, 99)
B2o = lsm(0.2 * st["oof_logp"].astype(np.float64) + 0.5 * st["tab_S3_oof"].astype(np.float64) + 0.3 * TA).astype(np.float32)
B2t = lsm(0.2 * st["test_logp"].astype(np.float64) + 0.5 * st["tab_S3_test"].astype(np.float64) + 0.3 * TT).astype(np.float32)
ref = lsm(0.2 * st["oof_logp"].astype(np.float64) + 0.5 * st["tab_S3_oof"].astype(np.float64) + 0.3 * st["TA_OOF"].astype(np.float64))
print(f"blend check: rebuilt kernel B2 vs stored max abs diff {np.abs(ref - st['B2_OOF']).max():.4f}; tile-level F1 kernel B2 {macro_f1(y, st['B2_OOF'].argmax(1)):.4f} -> new B2 {macro_f1(y, B2o.argmax(1)):.4f}")
G = os.path.join(W, "work", "good")
np.save(os.path.join(G, f"b2_{a.tag}_oof.npy"), B2o); np.save(os.path.join(G, f"b2_{a.tag}_test.npy"), B2t)
np.save(os.path.join(G, f"ta_{a.tag}_oof.npy"), TA); np.save(os.path.join(G, f"ta_{a.tag}_test.npy"), TT); print(f"saved [{time.time() - t0:.0f}s]")
