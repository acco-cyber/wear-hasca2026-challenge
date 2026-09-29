"""Louvain communities on per-subject kNN(video+P)+link graph -> purity ceilings and cluster decoders; ICM relabelling."""
import sys, time, itertools
import numpy as np
from common import *
from graphlib import knn_topk, build_W, louvain, icm

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS); lq = np.log(np.clip(Q, 1e-9, 1))
bid = true_bouts(y, rec, st)
E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
print("baseline", round(macro_f1(y, pred), 4))
subs = np.unique(sbj)
T = {}
t0 = time.time()
for s in subs:
    ii = np.flatnonzero(sbj == s); loc = -np.ones(n, np.int64); loc[ii] = np.arange(len(ii))
    sl = np.where(succ[ii] >= 0, loc[np.maximum(succ[ii], 0)], -1)
    for use_p in (0.0, 0.5):
        T[(s, use_p)] = (ii, sl) + knn_topk(E, P, ii, K=30, use_p=use_p)
print("knn done", round(time.time() - t0), "s", flush=True)


def evaluate(comm_all, tag):
    u, inv = np.unique(comm_all, return_inverse=True); nc = len(u)
    cnt_t = np.zeros((nc, N_CLS)); np.add.at(cnt_t, (inv, y), 1)
    ceil = macro_f1(y, cnt_t.argmax(1)[inv])
    cnt = np.zeros((nc, N_CLS)); np.add.at(cnt, (inv, pred), 1); pur = cnt.max(1) / cnt.sum(1); size = cnt.sum(1)
    sq = np.zeros((nc, N_CLS)); np.add.at(sq, inv, lq)
    bout_pur = np.mean([np.bincount(bid[inv == c]).max() / (inv == c).sum() for c in range(nc)]) if nc < 3000 else np.nan
    r = [f"{tag}: {nc} comms, med size {np.median(size):.0f}, TRUE-maj {ceil:.4f}"]
    best = None
    for how, lab in (("maj", cnt.argmax(1)), ("logq", sq.argmax(1))):
        for thr in (0.0, 0.7, 0.85):
            o = pred.copy(); use = (pur >= thr)[inv]; o[use] = lab[inv][use]; f = macro_f1(y, o)
            r.append(f"{how}@{thr} {f:.4f}")
    print(" | ".join(r), flush=True)


for use_p, k, beta, gamma in list(itertools.product((0.5,), (5, 10), (0.0, 1.0), (1.0, 3.0, 10.0, 30.0))) + [(0.0, 10, 1.0, 1.0), (0.0, 10, 1.0, 3.0)]:
    comm_all = np.zeros(n, np.int64); off = 0; t0 = time.time()
    for s in subs:
        ii, sl, nb, v = T[(s, use_p)]
        W = build_W(nb, v, k, 0.1, sl, score[ii], beta)
        c = louvain(W, gamma=gamma); comm_all[ii] = c + off; off += c.max() + 1
    evaluate(comm_all, f"p{use_p} k{k} beta{beta} gamma{gamma} ({time.time() - t0:.0f}s)")

print("\nICM Potts relabelling")
for use_p, k, beta, a in itertools.product((0.5,), (5, 10), (0.0, 1.0), (0.5, 1.0, 2.0)):
    out = pred.copy()
    for s in subs:
        ii, sl, nb, v = T[(s, use_p)]
        W = build_W(nb, v, k, 0.1, sl, score[ii], beta)
        # normalise W rows so total neighbour weight ~ 1 per node scale
        dg = np.asarray(W.sum(1)).ravel(); dg[dg == 0] = 1
        import scipy.sparse as sp
        Wn = sp.diags(1 / dg) @ W
        out[ii] = icm(Wn, lq[ii], a=a * 0.2, init=pred[ii])
    print(f"ICM p{use_p} k{k} beta{beta} a{a}: F1 {macro_f1(y, out):.4f} {per_fold(y, out, fold)}", flush=True)
