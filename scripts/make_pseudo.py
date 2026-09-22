#!/usr/bin/env python3
"""Generate pseudo-labels for test from the d1 decode pipeline (confidence-gated)."""
import numpy as np, pandas as pd, sys
from scipy.special import softmax, log_softmax

PROC = "/home/z/my-project/data/proc"
OUT = f"{PROC}/models"
Zte = np.load(f"{PROC}/video_te_pca.npy"); Zte /= (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)

Pte = np.zeros((12234, 19), np.float32)
for k in range(4):
    Pte += log_softmax(np.clip(np.load(f"{OUT}/p_te_{k}.npy"), 1e-9, 1))
Pte = softmax(Pte / 4, 1)
sys.path.insert(0, "/home/z/my-project/scripts")
from decode_v9c import propagate
P = propagate(Pte, Zte, k=12, alpha=0.25, iters=2)
L = np.log(np.clip(P, 1e-9, 1))
knn_te = np.load(f"{PROC}/knn_te.npy")
tm = pd.read_csv(f"{PROC}/../test/test_meta_data.csv"); te_sbj = tm.sbj_id.to_numpy()
knull = knn_te[:, 19]
for s in np.unique(te_sbj):
    m = te_sbj == s
    tgt = knull[m].mean()
    lo, hi = -8., 8.
    for _ in range(35):
        mid = (lo + hi) / 2
        r = (np.argmax(L[m] + np.array([mid] + [0] * 18), 1) == 0).mean()
        if r < tgt: lo = mid
        else: hi = mid
    L[m, 0] += (lo + hi) / 2
P = softmax(L, 1)
conf = P.max(1); yhat = P.argmax(1)
np.save(f"{PROC}/pseudo_y.npy", yhat.astype(np.int8))
np.save(f"{PROC}/pseudo_conf.npy", conf.astype(np.float32))
np.save(f"{PROC}/pseudo_P.npy", P.astype(np.float32))
for thr in [0.6, 0.7, 0.8, 0.9]:
    print(f"conf>{thr}: {(conf>thr).mean():.3f}")
print("class dist of pseudo:", np.bincount(yhat, minlength=19))
print("null rate:", (yhat == 0).mean().round(3))
