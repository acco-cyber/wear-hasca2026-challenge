"""(1) oracle decomposition: true-half prior on null rows only / activity rows only (predicted-null split is label-free)
(2) supervised (true half) ridge trained on null windows of a subject, applied to its activity windows: does the
    session direction transfer from null frames to activity frames?
(3) null-row accuracy of the cached label-free estimators vs true half"""
import os, sys
import numpy as np
from common import *
import methods as M
sm, P = load_oof(); y, sbj, rec, st, fold = sm["y"], sm["sbj"], sm["rec"], sm["start"], sm["fold"]
th = true_half(sm); a_ = y > 0
f0, pf0, pred0 = score(y, fold, sbj, P)
p_or = np.where(th == 1, 0.9, 0.1)
for nm, m in (("true-null rows", ~a_), ("activity rows", a_), ("pred-null rows (label-free split)", pred0 == 0),
              ("pred-activity rows", pred0 > 0)):
    p1 = np.where(m, p_or, 0.5); f, pf, _ = score(y, fold, sbj, apply_prior(P, p1, 2))
    print(f"oracle(0.9,a=2) on {nm}: {f:.4f} ({f-f0:+.4f})")
emb = M.l2n(np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32)); C = M.centred(emb, sbj)
accs = []
for s in np.unique(sbj):
    ii = np.flatnonzero(sbj == s); nn = ii[~a_[ii]]; aa = ii[a_[ii]]
    Xc = C[ii]; mu = Xc[~a_[ii]].mean(0)
    U, S, Vt = np.linalg.svd(C[nn] - C[nn].mean(0), full_matrices=False); B = Vt[:40]
    Zn = (C[nn] - mu) @ B.T; Za = (C[aa] - mu) @ B.T
    sd = Zn.std(0) + 1e-6; A = Zn / sd; t = np.where(th[nn] == 1, 1., -1.)
    beta = np.linalg.solve(A.T @ A + 0.1 * len(A) * np.eye(40), A.T @ t); b0 = (t - A @ beta).mean()
    sa = (Za / sd) @ beta + b0
    acc = np.mean((sa > 0) == (th[aa] == 1))
    # residual version: remove per-class (true) activity means first -> scene only
    Zr = Za.copy()
    for c in np.unique(y[aa]):
        m = y[aa] == c; Zr[m] -= Zr[m].mean(0)
    acc_r = np.mean((((Zr / sd) @ beta) > 0) == (th[aa] == 1))
    accs.append((acc, acc_r)); print(f"sbj {s:2d}: null->activity transfer acc {acc:.3f}", flush=True)
print("mean null->activity transfer acc", np.array(accs).mean(0).round(3))
cache = dict(np.load("oof_p1.npz"))
for nm, p1 in cache.items():
    print(f"{nm:18s} null-row acc vs true half {np.mean((p1[~a_] > 0.5) == (th[~a_] == 1)):.4f}  "
          f"pred-null rows {np.mean((p1[pred0 == 0] > 0.5) == (th[pred0 == 0] == 1)):.4f}")
