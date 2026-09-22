#!/usr/bin/env python3
"""WEAR v9 decoder:
- ensemble fold test probs (logit mean)
- label propagation among test windows (video-space kNN, iterated)
- null-logit calibration tuned on OOF (same propagation applied fold-honestly)
Usage: python3 decode_v9.py [alpha] [iters]"""
import sys, os
import numpy as np
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score

PROC = "/home/z/my-project/data/proc"
OUT = f"{PROC}/models"
NCLASS, NLIMB = 19, 4

def propagate(P, Z, k=12, alpha=0.25, iters=2, temp=0.05):
    """P: (n,19) probs; Z: (n,d) L2-normalized embedding. Returns smoothed probs.
    Neighbors: top-k most similar EXCLUDING self, weighted softmax(sim/temp)."""
    n = len(P)
    out = P.copy()
    for it in range(iters):
        Q = np.zeros_like(out)
        for st in range(0, n, 1024):
            en = min(n, st + 1024)
            sim = Z[st:en] @ Z.T
            sim[np.arange(en - st), np.arange(st, en)] = -2.0  # exclude self
            idx = np.argpartition(-sim, k, axis=1)[:, :k]
            rows = np.arange(en - st)[:, None]
            w = softmax(sim[rows, idx] / temp, axis=1)
            Q[st:en] = (w[:, :, None] * out[idx]).sum(1)
        out = (1 - alpha) * out + alpha * Q
    return out

def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    Ztr = np.load(f"{PROC}/video_tr_pca.npy")
    Ztr = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
    Zte = np.load(f"{PROC}/video_te_pca.npy")
    Zte = Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
    yw = np.load(f"{PROC}/train_y.npy").astype(np.int64)

    # ---- assemble OOF (window-level: average 4 limb-rows) ----
    oof = np.zeros((len(yw), 4, NCLASS), np.float32)
    filled = np.zeros(len(yw), bool)
    for k in range(4):
        idx = np.load(f"{OUT}/va_idx_{k}.npy")
        p = np.load(f"{OUT}/p_va_{k}.npy")
        win = idx // 4
        for j, w in enumerate(win):
            oof[w, idx[j] % 4] = p[j]
            filled[w] = True
    assert filled.all()
    P_oof_row = oof.reshape(-1, NCLASS)
    P_oof_win = softmax(np.log(np.clip(oof, 1e-9, 1)).mean(1), axis=1)  # logit-mean over limbs

    # ---- OOF honest propagation: per fold, pool = that fold's va windows ----
    vfw = np.load(f"{OUT}/va_idx_0.npy")  # not needed; rebuild fold assignment
    sys.path.insert(0, "/home/z/my-project/scripts")
    from train_v9 import folds_by_subject
    vfw = folds_by_subject()
    P_oof_sm = P_oof_win.copy()
    for k in np.unique(vfw):
        m = vfw == k
        P_oof_sm[m] = propagate(P_oof_win[m], Ztr[m], k=12, alpha=alpha, iters=iters)

    print(f"[OOF] base win-F1: {f1_score(yw, P_oof_win.argmax(1), average='macro'):.4f}")
    print(f"[OOF] smoothed win-F1 (a={alpha}, it={iters}): {f1_score(yw, P_oof_sm.argmax(1), average='macro'):.4f}")

    # ---- null calibration on OOF (smoothed) ----
    best = (0, 0.0)
    L = np.log(np.clip(P_oof_sm, 1e-9, 1))
    for delta in np.arange(-1.5, 1.51, 0.1):
        Lc = L.copy(); Lc[:, 0] += delta
        f1 = f1_score(yw, Lc.argmax(1), average="macro")
        if f1 > best[0]:
            best = (f1, delta)
    print(f"[OOF] best null delta={best[1]:+.1f} -> F1 {best[0]:.4f}")

    # ---- test predictions ----
    Pte = np.zeros((12234, NCLASS), np.float32)
    for k in range(4):
        Lk = log_softmax(np.clip(np.load(f"{OUT}/p_te_{k}.npy"), 1e-9, 1))
        Pte += Lk
    Pte = softmax(Pte / 4, axis=1)
    Pte_sm = propagate(Pte, Zte, k=12, alpha=alpha, iters=iters)
    Lte = np.log(np.clip(Pte_sm, 1e-9, 1))
    Lte[:, 0] += best[1]
    pred = Lte.argmax(1).astype(np.float64)
    sub = np.zeros((12234, 2), dtype=object)
    ids = np.arange(12234)
    np.savetxt("/home/z/my-project/download/sub_d1.csv",
               np.column_stack([ids, pred]), fmt="%d,%.1f",
               header="id,target_feature", comments="")
    print("null rate:", (pred == 0).mean().round(3), "| saved sub_d1.csv")
    print("test pred class dist:", np.bincount(pred.astype(int), minlength=19))

if __name__ == "__main__":
    main()
