"""Diagnostic (uses TRUE labels, not a method): is there block information in the video/IMU features that is NOT
activity identity?  Per subject, leave-one-family-out: train a ridge classifier block ~ features on two exercise
families (jog 1-5, stretch 6-10, strength 11-18), test on the held-out family.  ~0.5 = no scene signal."""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]
blk = np.zeros(19, int); blk[B1] = 1; blk[B2] = 2
fam = np.zeros(19, int); fam[1:6] = 1; fam[6:11] = 2; fam[11:19] = 3
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
y, rec, st, sbj = sm["y"], sm["rec"], sm["start"], sm["sbj"]
emb = np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32)
fo = np.load(os.path.join(KEEP, "feat_oof.npz"))
feats = {"emb": emb, "vpca": fo["vpca"].astype(np.float32), "vmot": fo["vmot"].astype(np.float32),
         "imu": np.nan_to_num(fo["imu"].astype(np.float32))}


def ridge_fit_pred(Xtr, ytr, Xte, lam=10.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    t = np.where(ytr == 1, 1.0, -1.0)
    # balance classes
    w = np.where(ytr == 1, 0.5 / max((ytr == 1).mean(), 1e-3), 0.5 / max((ytr == 2).mean(), 1e-3))
    Aw = A * w[:, None]
    G = A.T @ Aw + lam * len(A) / 100 * np.eye(A.shape[1])
    beta = np.linalg.solve(G, Aw.T @ t); b0 = np.average(t - A @ beta, weights=w)
    return B @ beta + b0


res = {k: [] for k in feats}
for s in np.unique(sbj):
    ii = np.flatnonzero((sbj == s) & (y > 0))
    line = f"sbj {s:2d}"
    for k, X in feats.items():
        accs = []
        for f in (1, 2, 3):
            tr = ii[fam[y[ii]] != f]; te = ii[fam[y[ii]] == f]
            sc = ridge_fit_pred(X[tr], blk[y[tr]], X[te])
            acc = np.mean((sc > 0) == (blk[y[te]] == 1)); accs.append(acc)
        res[k].append(accs); line += f"  {k}: " + "/".join(f"{a:.2f}" for a in accs)
    print(line, flush=True)
for k in feats:
    a = np.array(res[k]); print(k, "mean held-out-family block acc (jog/stretch/strength):", a.mean(0).round(3), "all", a.mean().round(3))
