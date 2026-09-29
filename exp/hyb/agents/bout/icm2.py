"""ICM refinement: stickiness bonus, emission source, nested fold selection, per-subject/per-class deltas"""
import itertools, pickle
import numpy as np, scipy.sparse as sp
from common import *
from graphlib import knn_topk, build_W, icm

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS); lq = np.log(np.clip(Q, 1e-9, 1)); lp = np.log(np.clip(P, 1e-9, 1))
base = macro_f1(y, pred); print("baseline", round(base, 4), per_fold(y, pred, fold))
E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
subs = np.unique(sbj); T = {}
for s in subs:
    ii = np.flatnonzero(sbj == s)
    for use_p in (0.5, 1.0, 2.0):
        T[(s, use_p)] = (ii,) + knn_topk(E, P, ii, K=12, use_p=use_p)
OH = np.zeros((n, N_CLS)); OH[np.arange(n), pred] = 1


def run_icm(use_p, k, a, b, src, tau=0.1, iters=10):
    out = pred.copy()
    for s in subs:
        ii, nb, v = T[(s, use_p)]
        W = build_W(nb, v, k, tau, -np.ones(len(ii), np.int64), None, 0.0)
        dg = np.asarray(W.sum(1)).ravel(); dg[dg == 0] = 1
        L = (lq if src == "q" else lp)[ii] * (a * 0.2) + b * OH[ii]
        out[ii] = icm(sp.diags(1 / dg) @ W, L, a=1.0, init=pred[ii], iters=iters)
    return out


grid = list(itertools.product((0.5, 1.0, 2.0), (3, 5, 7), (2.0, 3.0, 4.0, 6.0), (0.0, 0.1, 0.2), ("q", "p")))
outs = {}
for g in grid:
    o = run_icm(*g); outs[g] = o
    print(f"ICM use_p={g[0]} k={g[1]} a={g[2]} stick={g[3]} src={g[4]}: F1 {macro_f1(y, o):.4f} {per_fold(y, o, fold)} changed {np.mean(o != pred):.4f}", flush=True)
pickle.dump(outs, open("icm2_outs.pkl", "wb"))
# nested selection over folds
nest = pred.copy(); chosen = {}
for f in np.unique(fold):
    tr = fold != f
    best = max(grid, key=lambda g: macro_f1(y[tr], outs[g][tr])); chosen[int(f)] = best
    nest[fold == f] = outs[best][fold == f]
print("nested-selected:", chosen)
print(f"NESTED F1 {macro_f1(y, nest):.4f} (+{macro_f1(y, nest) - base:.4f}) {per_fold(y, nest, fold)}")
