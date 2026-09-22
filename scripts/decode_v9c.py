#!/usr/bin/env python3
"""WEAR v9 decoder v3: null-logit override toward video-kNN evidence + per-subject null matching.
Usage: python3 decode_v9c.py [--tune] | <lambda> <delta_mode>"""
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
    for _ in range(iters):
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

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def get_oof():
    yw = np.load(f"{PROC}/train_y.npy").astype(np.int64)
    oof = np.zeros((len(yw), 4, NCLASS), np.float32)
    filled = np.zeros(len(yw), bool)
    for k in range(4):
        idx = np.load(f"{OUT}/va_idx_{k}.npy")
        p = np.load(f"{OUT}/p_va_{k}.npy")
        oof[idx // 4, idx % 4] = p
        filled[idx // 4] = True
    Pw = softmax(np.log(np.clip(oof, 1e-9, 1)).mean(1), 1)
    return yw, Pw

def knn_null_mass(path):
    k = np.load(path)
    return k[:, 19]  # distance-weighted null prob from video neighbors

def apply_null_override(P, knull, lam, subject_ids=None, per_subject=False,
                        prop_alpha=0.25, prop_iters=2, Z=None, delta=0.0):
    L = log_softmax(np.clip(P, 1e-9, 1))
    ln = logit(knull)
    L[:, 0] = (1 - lam) * L[:, 0] + lam * ln
    if per_subject and subject_ids is not None:
        for s in np.unique(subject_ids):
            m = subject_ids == s
            tgt = knull[m].mean()
            # solve per-subject delta so that predicted null rate ~= tgt (bisection)
            lo, hi = -6.0, 6.0
            Ls = L[m]
            for _ in range(25):
                mid = (lo + hi) / 2
                r = (np.argmax(Ls + np.array([mid] + [0]*(NCLASS-1)), 1) == 0).mean()
                if r < tgt: lo = mid
                else: hi = mid
            L[m, 0] += (lo + hi) / 2
    elif delta != 0.0:
        L[:, 0] += delta
    P = softmax(L, 1)
    if Z is not None:
        P = propagate(P, Z, k=12, alpha=prop_alpha, iters=prop_iters)
    return P

if __name__ == "__main__":
    Ztr = np.load(f"{PROC}/video_tr_pca.npy"); Ztr /= (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
    Zte = np.load(f"{PROC}/video_te_pca.npy"); Zte /= (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
    yw, Pw = get_oof()
    kn_tr = knn_null_mass(f"{PROC}/knn_tr.npy")
    kn_te = knn_null_mass(f"{PROC}/knn_te.npy")
    sys.path.insert(0, "/home/z/my-project/scripts")
    from train_v9 import folds_by_subject
    vfw = folds_by_subject()
    sbw = np.load(f"{PROC}/train_sbj.npy").astype(np.int64)
    import pandas as pd
    te_sbj = pd.read_csv(f"{PROC}/../test/test_meta_data.csv").sbj_id.to_numpy()

    if "--tune" in sys.argv:
        best = (0,)
        for lam in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for ps in [False, True]:
                Lsm = np.zeros_like(Pw)
                for k in np.unique(vfw):
                    m = vfw == k
                    Lsm[m] = apply_null_override(Pw[m], kn_tr[m], lam,
                        subject_ids=sbw[m] if ps else None, per_subject=ps,
                        Z=Ztr[m])
                f1 = f1_score(yw, Lsm.argmax(1), average="macro")
                nr = (Lsm.argmax(1) == 0).mean()
                print(f"lam={lam:.1f} per subj={ps} OOF F1={f1:.4f} null={nr:.3f}", flush=True)
                if f1 > best[0]: best = (f1, lam, ps)
        print("BEST:", best)
        lam, ps = best[1], best[2]
    else:
        lam = float(sys.argv[1]); ps = len(sys.argv) > 2 and sys.argv[2] == "ps"

    Pte = np.zeros((12234, NCLASS), np.float32)
    for k in range(4):
        Pte += log_softmax(np.clip(np.load(f"{OUT}/p_te_{k}.npy"), 1e-9, 1))
    Pte = softmax(Pte / 4, 1)
    Pt = apply_null_override(Pte, kn_te, lam, subject_ids=te_sbj, per_subject=ps, Z=Zte)
    pred = Pt.argmax(1).astype(np.float64)
    np.savetxt("/home/z/my-project/download/sub_d1.csv",
               np.column_stack([np.arange(12234), pred]), fmt="%d,%.1f",
               header="id,target_feature", comments="")
    print(f"saved sub_d1.csv lam={lam} ps={ps} nullrate={(pred==0).mean():.3f}")
    print("per-sbj null rates:", {int(s): round((pred[te_sbj==s]==0).mean(),3) for s in np.unique(te_sbj)})
