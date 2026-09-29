"""small sweep of the best family (block-mass propagation on the within-subject video kNN graph)"""
import os, sys
import numpy as np
from common import *
import methods as M
sm, P = load_oof(); y, fold, sbj = sm["y"], sm["fold"], sm["sbj"]
emb = np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32)
l0 = np.load(os.path.join(KEEP, "links_L0.npz")); succ = l0["oof_succ"].astype(np.int64); score_ = l0["oof_score"].astype(np.float32)
f0, pf0, pred0 = score(y, fold, sbj, P); a_ = y > 0; tb = BLK[y]
cache = dict(np.load("oof_p1.npz"))
CF = {
    "kr_k5_a9": dict(graph="knn", feat="resid", knn=5, alpha=0.9, iters=50),
    "kr_k20_a9": dict(graph="knn", feat="resid", knn=20, alpha=0.9, iters=50),
    "kr_k10_a8": dict(graph="knn", feat="resid", knn=10, alpha=0.8, iters=30),
    "kr_k10_a95": dict(graph="knn", feat="resid", knn=10, alpha=0.95, iters=80),
    "kr_k10_a9_self": dict(graph="knn", feat="resid", knn=10, alpha=0.9, iters=50, self_out=True),
    "kraw_k10_a9": dict(graph="knn", feat="raw", knn=10, alpha=0.9, iters=50),
    "br_k10_a9": dict(graph="both", feat="resid", knn=10, alpha=0.9, iters=50, w_link=0.3),
}
for nm, kw in CF.items():
    p1 = M.m_prop(P, emb, sbj, succ, score_, **kw); cache[nm] = p1
    acc = np.mean((p1[a_] > 0.5) == (tb[a_] == 1))
    res = []
    for mk, m in (("all", None), ("predact", pred0 > 0)):
        for a in (0.5, 1.0):
            f, pf, _ = score(y, fold, sbj, apply_prior(P, p1, a, "keepnull", m)); d = np.array(pf) - pf0
            res.append(f"{mk}/a{a}: {f-f0:+.4f} [{' '.join(f'{x:+.4f}' for x in d)}]")
    print(f"{nm:16s} act-acc {acc:.4f} | " + " | ".join(res), flush=True)
np.savez("oof_p1.npz", **cache)
