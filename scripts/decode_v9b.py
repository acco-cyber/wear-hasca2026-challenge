#!/usr/bin/env python3
"""WEAR v9 decoder v2: LGBM ensemble + video-kNN logit blend + propagation + null calib.
All hyperparams tuned on fold-safe OOF, then applied to test.
Usage: python3 decode_v9b.py [w] [delta] [--tune]"""
import sys, os
import numpy as np
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score

PROC = "/home/z/my-project/data/proc"
OUT = f"{PROC}/models"
NCLASS = 19
EPS = 1e-7

def propagate(P, Z, k=12, alpha=0.25, iters=2, temp=0.05):
    n = len(P); out = P.copy()
    for it in range(iters):
        Q = np.zeros_like(out)
        for st in range(0, n, 1024):
            en = min(n, st + 1024)
            sim = Z[st:en] @ Z.T
            sim[np.arange(en - st), np.arange(st, en)] = -2.0
            idx = np.argpartition(-sim, k, axis=1)[:, :k]
            rows = np.arange(en - st)[:, None]
            w = softmax(sim[rows, idx] / temp, axis=1)
            Q[st:en] = (w[:, :, None] * out[idx]).sum(1)
        out = (1 - alpha) * out + alpha * Q
    return out

def get_oof_and_knn():
    yw = np.load(f"{PROC}/train_y.npy").astype(np.int64)
    oof = np.zeros((len(yw), 4, NCLASS), np.float32)
    knnr = np.zeros((len(yw), 4, 40), np.float32)
    filled = np.zeros(len(yw), bool)
    for k in range(4):
        idx = np.load(f"{OUT}/va_idx_{k}.npy")
        p = np.load(f"{OUT}/p_va_{k}.npy")
        w = idx // 4
        oof[w, idx % 4] = p
        filled[w] = True
    knn_tr = np.load(f"{PROC}/knn_tr.npy")
    knnr = np.repeat(knn_tr[:, None, :], 4, 1)  # window-level knn feats replicated
    Pw = softmax(np.log(np.clip(oof, 1e-9, 1)).mean(1), 1)
    return yw, Pw, knn_tr

def apply_pipeline(P_model, knn_w, Z, w_blend, delta, prop=(0.25, 2)):
    """P_model: (n,19); knn_w: (n,40) window-level knn feats"""
    L = log_softmax(np.clip(P_model, 1e-9, 1))
    Lk = np.log(np.clip(knn_w[:, 19:38], EPS, 1) + EPS)  # distance-weighted hist
    Lb = (1 - w_blend) * L + w_blend * Lk
    P = softmax(Lb, 1)
    P = propagate(P, Z, k=12, alpha=prop[0], iters=prop[1])
    L = np.log(np.clip(P, EPS, 1))
    L[:, 0] += delta
    return L

if __name__ == "__main__":
    Ztr = np.load(f"{PROC}/video_tr_pca.npy"); Ztr /= (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
    Zte = np.load(f"{PROC}/video_te_pca.npy"); Zte /= (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
    yw, Pw_oof, knn_tr = get_oof_and_knn()
    sys.path.insert(0, "/home/z/my-project/scripts")
    from train_v9 import folds_by_subject
    vfw = folds_by_subject()

    if "--tune" in sys.argv:
        # OOF-honest: propagate within each fold's va pool
        results = []
        for w_blend in [0.0, 0.15, 0.3, 0.45, 0.6]:
            Lsm = np.zeros_like(Pw_oof)
            for k in np.unique(vfw):
                m = vfw == k
                Lsm[m] = apply_pipeline(Pw_oof[m], knn_tr[m], Ztr[m], w_blend, 0.0)
            for delta in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]:
                pred = (Lsm + np.eye(NCLASS)[0] * 0).copy()
                Lc = Lsm.copy(); Lc[:, 0] += delta
                f1 = f1_score(yw, Lc.argmax(1), average="macro")
                nr = (Lc.argmax(1) == 0).mean()
                results.append((f1, w_blend, delta, nr))
                print(f"w={w_blend:.2f} d={delta:+.2f} F1={f1:.4f} nullrate={nr:.3f}", flush=True)
        results.sort(reverse=True)
        f1, w_blend, delta, nr = results[0]
        print(f"BEST OOF: w={w_blend} d={delta} F1={f1:.4f} null={nr:.3f}")
    else:
        w_blend = float(sys.argv[1]); delta = float(sys.argv[2])

    # test predictions
    Pte = np.zeros((12234, NCLASS), np.float32)
    for k in range(4):
        Pte += log_softmax(np.clip(np.load(f"{OUT}/p_te_{k}.npy"), 1e-9, 1))
    Pte = softmax(Pte / 4, 1)
    knn_te = np.load(f"{PROC}/knn_te.npy")
    Lte = apply_pipeline(Pte, knn_te, Zte, w_blend, delta)
    pred = Lte.argmax(1).astype(np.float64)
    np.savetxt("/home/z/my-project/download/sub_d1.csv",
               np.column_stack([np.arange(12234), pred]), fmt="%d,%.1f",
               header="id,target_feature", comments="")
    print(f"saved sub_d1.csv w={w_blend} d={delta} nullrate={(pred==0).mean():.3f}")
