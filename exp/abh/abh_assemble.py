"""Assemble finished GroupKFold(3) folds (abh_cv3/fold{k}.npz) -> row-level OOF macro-F1 (notebook metric, labels y_c),
single-limb OOF F1 (exp/base common.eval_single, our standard), abh_cv3/oof.npy (N,4,19; NaN where limb IMU is NaN or
fold missing) and, when all 3 folds exist, abh_cv3/test.npy = mean of the 3 fold models (the notebook's own test_probs)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, r"E:\Claude code\wear\exp\base")
from common import eval_single

HERE = os.path.dirname(os.path.abspath(__file__)); CV = os.path.join(HERE, "abh_cv3"); CACHE = os.path.join(HERE, "cache")
tr = np.load(os.path.join(CACHE, "train.npz")); y = tr["y"]; nan = tr["nan"]; N = len(y)
Y = np.tile(y, 4)
oof = np.full((N, 4, 19), np.nan, np.float32); tests = []; vi_all = []; pv_all = []
for k in range(3):
    p = os.path.join(CV, f"fold{k}.npz")
    if not os.path.exists(p): print(f"fold {k}: missing"); continue
    d = np.load(p); oof[d["rows"]] = d["oof"]; tests.append(d["test"]); vi_all.append(d["val_idx"]); pv_all.append(d["pv"])
    print(f"fold {k}: rows {len(d['rows'])} best_iter {int(d['best_iter'])} row-level macro-F1 {float(d['f1']):.4f}")
if vi_all:
    vi = np.concatenate(vi_all); pv = np.concatenate(pv_all)
    print(f"row-level OOF macro-F1 over done folds ({len(vi)} rows): {f1_score(Y[vi], pv.argmax(1), average='macro'):.4f}"
          f"  (notebook: 0.5871 over all 3 folds; fold F1s 0.5805/0.5930/0.5841)")
    oof_nan = oof.copy(); oof_nan[nan] = np.nan
    np.save(os.path.join(CV, "oof.npy"), oof_nan)
    if len(vi_all) == 3: eval_single(oof_nan, name="abh cv3 single-limb")
if len(tests) == 3:
    T = np.mean(tests, 0).astype(np.float32); np.save(os.path.join(CV, "test.npy"), T); print("wrote abh_cv3/test.npy", T.shape)
