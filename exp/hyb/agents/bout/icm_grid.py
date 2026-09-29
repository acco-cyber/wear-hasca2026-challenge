"""(1) oracle partition of the gap; (2) ICM (hard-label Potts relabelling on kNN graph) grid, per fold"""
import itertools, time
import numpy as np, scipy.sparse as sp
from common import *
from graphlib import knn_topk, build_W, icm

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS); lq = np.log(np.clip(Q, 1e-9, 1))
bid = true_bouts(y, rec, st)
base = macro_f1(y, pred); print("baseline", round(base, 4), per_fold(y, pred, fold))
# (1) oracle partition
u, inv = np.unique(bid, return_inverse=True)
bmaj = np.array([np.bincount(pred[inv == b], minlength=N_CLS).argmax() for b in range(len(u))])
blen = np.bincount(inv); pos = np.zeros(n, np.int64); edge = np.zeros(n, np.int64)
for b in range(len(u)):
    ii = np.flatnonzero(inv == b); ii = ii[np.argsort(st[ii])]; L = len(ii); p_ = np.arange(L); edge[ii] = np.minimum(p_, L - 1 - p_)
err = pred != y; whole = err & (bmaj[inv] != y); inb = err & ~whole; bnd = inb & (edge <= 5); ins = inb & (edge > 5)
for nm, m in (("fix whole-bout", whole), ("fix boundary<=5s", bnd), ("fix inside", ins), ("fix boundary+inside", inb)):
    o = pred.copy(); o[m] = y[m]; print(f"oracle {nm}: F1 {macro_f1(y, o):.4f} (+{macro_f1(y, o) - base:.4f})")
# per subject class-count variability (true)
cnts = np.array([[np.sum((sbj == s) & (y == c)) for c in range(1, N_CLS)] for s in np.unique(sbj)])
print("true windows per exercise per subject: median", np.median(cnts), "q10/q90", np.quantile(cnts, [.1, .9]), "cv within subject", np.round(np.median(cnts.std(1) / cnts.mean(1)), 3))
# (2) ICM grid
E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
subs = np.unique(sbj); T = {}
for s in subs:
    ii = np.flatnonzero(sbj == s); loc = -np.ones(n, np.int64); loc[ii] = np.arange(len(ii))
    sl = np.where(succ[ii] >= 0, loc[np.maximum(succ[ii], 0)], -1)
    for use_p in (0.0, 0.5, 1.0):
        T[(s, use_p)] = (ii, sl) + knn_topk(E, P, ii, K=20, use_p=use_p)
res = []
for use_p, k, beta, a, mutual in itertools.product((0.0, 0.5, 1.0), (3, 5, 7, 10), (0.0, 0.5), (1.5, 2.0, 3.0, 4.0, 6.0), (False,)):
    out = pred.copy()
    for s in subs:
        ii, sl, nb, v = T[(s, use_p)]
        W = build_W(nb, v, k, 0.1, sl, score[ii], beta, mutual=mutual)
        dg = np.asarray(W.sum(1)).ravel(); dg[dg == 0] = 1
        out[ii] = icm(sp.diags(1 / dg) @ W, lq[ii], a=a * 0.2, init=pred[ii])
    f = macro_f1(y, out); pf = per_fold(y, out, fold)
    res.append((f, use_p, k, beta, a)); print(f"ICM p{use_p} k{k} beta{beta} a{a}: F1 {f:.4f} (+{f - base:.4f}) {pf} changed {np.mean(out != pred):.4f}", flush=True)
res.sort(reverse=True); print("top", res[:8])
