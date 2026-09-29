"""Diagnostic (true labels/time used ONLY for scoring): can the two halves (sessions/blocks) of a subject be separated
in activity-invariant directions?  Per subject:
  feature variants: cent = subject-centred emb; projM = centred emb with the top-M principal directions of the OTHER
  subjects' centred embeddings projected out (shared activity variation removed, subject-specific scene kept);
  resid = emb minus soft class-centroid reconstruction (P); lda = centred emb with train-class-mean span removed.
  scores: k-means(2) purity on activity windows (perm-free, best of 2 assignments) and on true-null windows;
  supervised blocked-CV ridge on true-null windows (upper bound of a 'session' signal in null frames)."""
import os, sys
import numpy as np
from common import *
import methods as M
sm, P = load_oof(); y, sbj, rec, st = sm["y"], sm["sbj"], sm["rec"], sm["start"]
emb = M.l2n(np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32))
fo = dict(np.load(os.path.join(KEEP, "feat_oof.npz")))
th = true_half(sm); a_ = y > 0
C = M.centred(emb, sbj)
R = M.residual_emb(emb, P, sbj)
subs = np.unique(sbj)
# other-subject PCA bases
bases = {}
for s in subs:
    o = sbj != s; Xo = C[o][::3]
    U, S, Vt = np.linalg.svd(Xo - Xo.mean(0), full_matrices=False); bases[s] = Vt[:200]
# train class-mean span (other subjects, true labels -> transferable to test)
cm_basis = {}
for s in subs:
    o = sbj != s; Mu = np.stack([C[o & (y == c)].mean(0) for c in range(19)])
    U, S, Vt = np.linalg.svd(Mu - Mu.mean(0), full_matrices=False); cm_basis[s] = Vt[:18]


def ridge_blocked(X, lab, tpos, lam=10.0, k=5):
    """blocked CV by time position quantile inside each half"""
    out = np.zeros(len(X))
    q = np.zeros(len(X), int)
    for h in (1, 2):
        m = lab == h; r = np.argsort(np.argsort(tpos[m])); q[m] = (r * k // max(m.sum(), 1))
    for f in range(k):
        tr, te = q != f, q == f
        if len(np.unique(lab[tr])) < 2 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = (X[tr] - mu) / sd; t = np.where(lab[tr] == 1, 1., -1.)
        beta = np.linalg.solve(A.T @ A + lam * len(A) / 100 * np.eye(A.shape[1]), A.T @ t)
        out[te] = ((X[te] - mu) / sd) @ beta + (t - A @ beta).mean()
    return np.mean((out > 0) == (lab == 1))


variants = ["cent", "proj20", "proj50", "proj100", "proj200", "resid", "lda"]
res = {v: [] for v in variants}
for s in subs:
    ii = np.flatnonzero(sbj == s); tpos = st[ii] + rec[ii] * 10 ** 7
    line = f"sbj {s:2d} n {len(ii):5d}"
    for v in variants:
        if v == "cent":
            X = C[ii]
        elif v.startswith("proj"):
            B = bases[s][:int(v[4:])]; X = C[ii] - (C[ii] @ B.T) @ B
        elif v == "resid":
            X = R[ii]
        else:
            B = cm_basis[s]; X = C[ii] - (C[ii] @ B.T) @ B
        Z = M.pca(X, 20)
        lab = M.kmeans(Z, 2, seed=int(s), restarts=2)
        aa = a_[ii]; tt = th[ii]
        acc_a = np.mean((lab[aa] == 0) == (tt[aa] == 1)); acc_a = max(acc_a, 1 - acc_a)
        nn = ~aa; acc_n = np.mean((lab[nn] == 0) == (tt[nn] == 1)); acc_n = max(acc_n, 1 - acc_n)
        sup = ridge_blocked(M.pca(X[nn], 40), tt[nn], tpos[nn])
        res[v].append((acc_a, acc_n, sup)); line += f" | {v}: {acc_a:.2f}/{acc_n:.2f}/{sup:.2f}"
    print(line, flush=True)
print("mean over subjects (kmeans2 purity on activity / on null, supervised blocked ridge on null):")
for v in variants:
    print(f"  {v:8s}", np.array(res[v]).mean(0).round(3))
# IMU gravity per limb on true-null windows: supervised blocked nearest-centroid
names = __import__("json").load(open(os.path.join(KEEP, "imu_names.json")))
gi = [names.index(k) for k in ("grav_x", "grav_y", "grav_z")]
sen = sm["sensor"]; accs = []
for s in subs:
    ii = np.flatnonzero((sbj == s) & (y == 0)); r = []
    for L in range(4):
        jj = ii[sen[ii] == L]
        r.append(ridge_blocked(fo["imu"][jj][:, gi].astype(np.float64), th[jj], (st[jj] + rec[jj] * 10 ** 7).astype(float), lam=1.0))
    accs.append(r)
print("IMU gravity (per limb, null windows) supervised blocked acc: mean", np.array(accs).mean(0).round(3), "per subject", np.array(accs).mean(1).round(2))
