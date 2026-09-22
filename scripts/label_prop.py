#!/usr/bin/env python3
"""Label propagation over joint video kNN graph: train (labeled) + pool (model probs).
OOF validation: per fold, pool = va windows of that fold with fold-OOF probs as init.
Usage: python3 label_prop.py [--tune]"""
import sys
import numpy as np
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score

PROC = "/home/z/my-project/data/proc"
OUT = f"{PROC}/models"
NCLASS = 19

def knn_graph_apply(P_db, Y_or_P_db, P_q, Z_db, Z_q, k=15, alpha=0.55, iters=6, temp=0.05,
                    self_loop=0.0):
    """Iterated propagation: P_q <- alpha * W@P_db_init + (1-alpha)*P_q0, where W built from
    query->db cosine kNN. db side gets its own self-consistency via mutual kNN within db? 
    Simplified: joint graph with db fixed as anchors, queries propagate among themselves too:
    full joint: nodes = db + q; iterate P over all nodes with anchors clamped."""
    n_db, n_q = len(P_db), len(P_q)
    Z_all = np.concatenate([Z_db, Z_q], 0)
    P = np.concatenate([P_db, P_q], 0).copy()  # (n_db+n_q, 19) probs
    N = n_db + n_q
    anchor_w = np.concatenate([np.ones(n_db, np.float32), np.full(n_q, self_loop, np.float32)])
    # build sparse-ish kNN index per node (chunked)
    idx_all = np.zeros((N, k), np.int64)
    w_all = np.zeros((N, k), np.float32)
    for st in range(0, N, 1024):
        en = min(N, st + 1024)
        sim = Z_all[st:en] @ Z_all.T
        sim[np.arange(en - st), np.arange(st, en)] = -2.0
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(en - st)[:, None]
        w_all[st:en] = softmax(sim[rows, idx] / temp, axis=1)
        idx_all[st:en] = idx
    for it in range(iters):
        Q = np.zeros_like(P)
        for st in range(0, N, 2048):
            en = min(N, st + 2048)
            Q[st:en] = (w_all[st:en, :, None] * P[idx_all[st:en]]).sum(1)
        P = anchor_w[:, None] * P + (1 - anchor_w[:, None]) * (alpha * Q + (1 - alpha) * P)
    return P[n_db:]

def main_tune():
    yw = np.load(f"{PROC}/train_y.npy").astype(np.int64)
    knn_tr = np.load(f"{PROC}/knn_tr.npy")
    Ztr = np.load(f"{PROC}/video_tr_pca.npy"); Ztr /= (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
    oof = np.zeros((len(yw), 4, NCLASS), np.float32)
    for k in range(4):
        idx = np.load(f"{OUT}/va_idx_{k}.npy")
        oof[idx // 4, idx % 4] = np.load(f"{OUT}/p_va_{k}.npy")
    Pw = softmax(np.log(np.clip(oof, 1e-9, 1)).mean(1), 1)
    sys.path.insert(0, "/home/z/my-project/scripts")
    from train_v9 import folds_by_subject
    vfw = folds_by_subject()
    for k in np.unique(vfw):
        m = vfw == k  # va windows = query pool
        d = ~m        # train windows = anchors
        # per fold: db = train-fold windows (labeled one-hot), q = va windows (model probs)
        Y_db = np.eye(NCLASS, dtype=np.float32)[yw[d]]
        for alpha in [0.4, 0.6, 0.8]:
            Pq = knn_graph_apply(Y_db, None, Pw[m], Ztr[d], Ztr[m],
                                 k=15, alpha=alpha, iters=6)
            f1 = f1_score(yw[m], Pq.argmax(1), average="macro")
            print(f"fold{k} alpha={alpha}: pool F1={f1:.4f} (base {f1_score(yw[m], Pw[m].argmax(1), average='macro'):.4f})", flush=True)
        break  # timing check on fold 0 first

if __name__ == "__main__":
    main_tune()
