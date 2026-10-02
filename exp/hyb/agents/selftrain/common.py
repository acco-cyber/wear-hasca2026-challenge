"""Shared helpers for the cross-fitted self-training experiments (tabular experts S3/T adapted to held-out subjects)."""
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, breadth_first_order
HYB = r"E:\Claude code\wear\exp\hyb"; W = r"E:\Claude code\wear"
sys.path.insert(0, HYB)
import hanbat_stack as H
import graph_lab as G
from hanbat_stack import KEEP, HB, N_CLS, TRAIN_SETS, TEST_CFGS, macro_f1, lsm
HERE = os.path.dirname(os.path.abspath(__file__))
TAB_PARAMS = dict(H.TAB_PARAMS); TAB_PARAMS["num_threads"] = 4
ROUNDS = dict(H.TAB_ROUNDS)
BLEND = (("S3", 0.5), ("T", 0.3))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_meta():
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    return sm


def features(split):
    """cached S3 / T design matrices (exactly hanbat_stack.tab_build, no extras)"""
    pS, pT = os.path.join(HERE, f"X_S3_{split}.npy"), os.path.join(HERE, f"X_T_{split}.npy")
    if not (os.path.exists(pS) and os.path.exists(pT)):
        d, t, fo, ft, names = H.load_all()
        dd, ff = (d, fo) if split == "oof" else (t, ft)
        np.save(pS, H.tab_build("S3", dd, ff, names, None)); np.save(pT, H.tab_build("T", dd, ff, names, None))
    return {"S3": np.load(pS, mmap_mode="r"), "T": np.load(pT, mmap_mode="r")}


def ours_oof(n):
    rows = np.load(os.path.join(HYB, "rows.npz")); o2t = rows["ours_to_theirs"]
    R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb")); ours = np.full(n, -1, np.int64)
    for s, d in R.items():
        ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
    has = ours >= 0
    return np.where(has, ours, 0), has


def extra_links_oof(n):
    sys.path.insert(0, os.path.join(W, "exp", "transductive")); from tlib import load_structs
    o2t = np.load(os.path.join(HYB, "rows.npz"))["ours_to_theirs"]
    s2 = np.full(n, -1, np.int64); c2 = np.full(n, -50.0, np.float32)
    for which in ("eval", "extra", "extra2"):
        for s, st in load_structs(which).items():
            a0, m = int(st["a"]), int(st["n"]); su = np.asarray(st["succ0"]); sc = np.asarray(st["sc"], np.float32)
            ok = (su >= 0) & (sc >= -6.0); rows_t = o2t[a0 + np.arange(m)]
            s2[rows_t[ok]] = o2t[a0 + su[ok]]; c2[rows_t[ok]] = sc[ok]
    return s2, c2


def extra_links_test(n):
    struct = pickle.load(open(os.path.join(W, "work", "test_structure.pkl"), "rb"))
    s2 = np.full(n, -1, np.int64); c2 = np.full(n, -50.0, np.float32)
    for s, st in struct.items():
        idx = np.asarray(st["idx"]); su = np.asarray(st["succ0"]); sc = np.asarray(st["sc"], np.float32); ok = (su >= 0) & (sc >= -6.0)
        s2[idx[ok]] = idx[su[ok]]; c2[idx[ok]] = sc[ok]
    return s2, c2


def graph_dd_oof(logp, sm, extra=True, _cache={}):
    l0 = np.load(os.path.join(KEEP, "links_L0.npz"))
    if "emb" not in _cache:
        _cache["emb"] = np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32)
        _cache["xl"] = extra_links_oof(len(sm["y"]))
    dd = dict(logp=logp.astype(np.float32), emb=_cache["emb"], grp=sm["sbj"], sbj=sm["sbj"], succ=l0["oof_succ"].astype(np.int64),
              score=l0["oof_score"].astype(np.float32), sets=TRAIN_SETS)
    if extra:
        s2, c2 = _cache["xl"]; dd.update(succ2=s2, score2=c2, xl_w=1.0, xl_b=-2.0, xlinks=[])
    return dd


def recipe(dd, ours, has, prior=0.3, counts=0.3, gate="0.55:top2"):
    """graph_lab.run without printing; returns final labels, ungated labels, calibrated Q, graph P, gate mask"""
    sbj = dd["sbj"]
    cover = {int(s): bool(has[sbj == s].all()) for s in np.unique(sbj)}
    targets = G.make_targets(sbj, dd["sets"], np.where(has, ours, 0), counts, cover)
    OH = None
    if prior > 0:
        OH = np.full((len(sbj), N_CLS), 1.0 / N_CLS); eps = 0.1
        OH[has] = eps / N_CLS; OH[np.flatnonzero(has), ours[has]] += 1 - eps
    P = G.graph_P2(dd, TEST_CFGS, targets, OH, prior)
    base, Q = G.finish2(P, sbj, targets)
    lab, g = base, np.zeros(len(base), bool)
    if gate:
        tau, rule = (gate.split(":") + ["plain"])[:2]
        lab, g = G.apply_gate(base, Q, ours, has, tau, rule, sbj=sbj)
    return lab, base, Q, P, g


def plain_graph(logp, sm):
    l0 = np.load(os.path.join(KEEP, "links_L0.npz"))
    dg = dict(logp=logp, emb=graph_dd_oof(logp, sm, extra=False)["emb"], grp=sm["sbj"], sbj=sm["sbj"],
              succ=l0["oof_succ"].astype(np.int64), score=l0["oof_score"].astype(np.float32), sets=TRAIN_SETS)
    P = H.graph_P(dg, TEST_CFGS); return H.finish(P, dg)


def per_fold(y, lab, fold):
    return [macro_f1(y[fold == f], lab[fold == f]) for f in range(5)]


def base_parts():
    """baseline (no self-training) OOF S3 / T logp: T from cache, S3 recovered from the saved tab blend"""
    bl = np.load(os.path.join(KEEP, "blend.npz")); win = bl["oof_logp"].astype(np.float64)
    LB = np.load(os.path.join(HB, "cv_base_oof_logp_b.npy")).astype(np.float64)
    T = np.load(os.path.join(HB, "cache_T_oof.npy")).astype(np.float64)
    S3 = lsm((LB - 0.2 * win - 0.3 * T) / 0.5)
    return win.astype(np.float32), S3.astype(np.float32), T.astype(np.float32), LB.astype(np.float32)


def tab_blend(win, S3, T):
    return H.tab_blend(win, [(S3, 0.5), (T, 0.3)])


def cross_groups(succ_list, n, sbj, cap=32):
    """connected components of the union of link sets; components larger than cap are cut into BFS-order chunks"""
    src, dst = [], []
    for su in succ_list:
        m = su >= 0; src.append(np.flatnonzero(m)); dst.append(su[m])
    src = np.concatenate(src); dst = np.concatenate(dst)
    keep = sbj[src] == sbj[dst]; src, dst = src[keep], dst[keep]
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)); A = (A + A.T).tocsr()
    nc, comp = connected_components(A, directed=False)
    grp = comp.copy(); nxt = nc
    sizes = np.bincount(comp)
    for c in np.flatnonzero(sizes > cap):
        members = np.flatnonzero(comp == c)
        order = breadth_first_order(A, members[0], directed=False, return_predecessors=False)
        for j in range(0, len(order), cap):
            grp[order[j:j + cap]] = nxt; nxt += 1
    return grp


def assign_parts(rows, grp, K, seed):
    """rows: indices to split; groups kept together; greedy balance over a random group order (per subject handled by caller)"""
    rng = np.random.default_rng(seed)
    g = grp[rows]; ug, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    order = rng.permutation(len(ug)); load = np.zeros(K); part_of = np.zeros(len(ug), int)
    for i in order:
        k = int(np.argmin(load + rng.random(K) * 1e-6)); part_of[i] = k; load[k] += cnt[i]
    return part_of[inv]
