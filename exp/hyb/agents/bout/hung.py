"""one-of-each constraints: (a) oracle true bouts, (b) Louvain communities.
 H1 'strict': Hungarian 18 classes -> 18 largest-evidence clusters, all other clusters null
 H2 'cover' : Hungarian guarantees each exercise gets >=1 cluster (clusters >= min size), others take their argmax"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *
from graphlib import knn_topk, build_W, louvain

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS); lq = np.log(np.clip(Q, 1e-9, 1))
bid = true_bouts(y, rec, st)
base = macro_f1(y, pred); print("baseline", round(base, 4))


def decode(comm, mode, min_size=10, keep_pur=None):
    out = pred.copy()
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); u, inv = np.unique(comm[ii], return_inverse=True); nc = len(u)
        S = np.zeros((nc, N_CLS)); np.add.at(S, inv, lq[ii]); size = np.bincount(inv, minlength=nc)
        cnt = np.zeros((nc, N_CLS)); np.add.at(cnt, (inv, pred[ii]), 1); pur = cnt.max(1) / size
        lab = S.argmax(1)
        if mode == "strict":
            lab = np.zeros(nc, np.int64)
            big = np.flatnonzero(size >= min_size)
            r, c = linear_sum_assignment(-(S[big][:, 1:] - S[big][:, [0]]))   # gain over null
            lab[big[r]] = c + 1
        elif mode == "cover":
            big = np.flatnonzero(size >= min_size)
            # cost: loss of forcing class c on cluster instead of its argmax
            G = S[big][:, 1:] - S[big].max(1, keepdims=True)
            r, c = linear_sum_assignment(-G); lab = lab.copy(); lab[big[r]] = c + 1
        o = lab[inv]
        if keep_pur is not None:
            keep = pur[inv] < keep_pur; o[keep] = pred[ii][keep]
        out[ii] = o
    return out


print("oracle true bouts:")
for mode in ("argmax", "strict", "cover"):
    for ms in (10, 30):
        o = decode(bid, mode, ms); print(f"  {mode} min{ms}: F1 {macro_f1(y, o):.4f} {per_fold(y, o, fold)}")
# missing classes under oracle bout majority
E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
comm = np.zeros(n, np.int64); off = 0
for gamma in (1.0, 3.0):
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); nb, v = knn_topk(E, P, ii, K=10, use_p=0.5)
        W = build_W(nb, v, 10, 0.1, -np.ones(len(ii), np.int64), score[ii], 0.0)
        c = louvain(W, gamma=gamma); comm[ii] = c + off; off += c.max() + 1
    print(f"Louvain gamma {gamma}:")
    for mode in ("argmax", "strict", "cover"):
        for kp in (None, 0.8):
            o = decode(comm, mode, 10, kp); print(f"  {mode} keep_pur<{kp}: F1 {macro_f1(y, o):.4f} {per_fold(y, o, fold)}")
