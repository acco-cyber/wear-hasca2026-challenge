"""Shared utilities for the links experiments (copied/adapted from src/chain.py, src/decode.py, src/sim_decode.py)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
import sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment

W = r"E:\Claude code\wear"
DATA = os.path.join(W, "data"); PREP = os.path.join(DATA, "prep"); WORK = os.path.join(W, "work"); EXP = os.path.join(W, "exp", "links")
sys.path.insert(0, os.path.join(W, "src"))
from decode import viterbi_chains, calibrate_counts, build_graph, graph_smooth  # noqa: E402  (read-only reference)

EVAL = ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"]
OLD_FIT = ["sbj_1", "sbj_3", "sbj_7", "sbj_12", "sbj_16", "sbj_19"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]

def load_prep():
    meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    imu = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r")
    vid = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
    return meta, imu, vid

def session_slices(meta):
    return {s: (g.index.min(), g.index.max() + 1) for s, g in meta.groupby("session", sort=False)}

def blend_oof(w2=0.2):
    o1 = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy")); o2 = np.load(os.path.join(WORK, "fusion_v1", "oof.npy"))
    o = np.exp((1 - w2) * np.log(np.clip(o1, 1e-6, 1)) + w2 * np.log(np.clip(o2, 1e-6, 1)))
    o /= np.nansum(o, 2, keepdims=True); return o

def blend_test(w2=0.2, p1=None, p2=None):
    p1 = p1 or os.path.join(WORK, "lgbm_v1", "test.npy"); p2 = p2 or os.path.join(WORK, "fusion_v1", "test.npy")
    L = (1 - w2) * np.log(np.clip(np.load(p1).astype(np.float64), 1e-6, 1)) + w2 * np.log(np.clip(np.load(p2).astype(np.float64), 1e-6, 1))
    P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)

def assignment_from(cand, lo, n):
    L = np.full((n, n), -50.0, np.float32); valid = cand >= 0
    rows = np.repeat(np.arange(n)[:, None], cand.shape[1], 1)
    L[rows[valid], cand[valid]] = lo[valid]
    np.fill_diagonal(L, -1e4)
    r, c = linear_sum_assignment(-L)
    return c.copy(), L[r, c].copy(), L

def cut(succ, sc, L, min_logodds):
    succ = succ.copy(); n = len(succ)
    succ[sc < min_logodds] = -1
    visited = np.zeros(n, bool)
    for st in range(n):
        if visited[st] or succ[st] < 0: continue
        path = []; x = st
        while x >= 0 and not visited[x]:
            visited[x] = True; path.append(x); x = succ[x]
        if x >= 0 and x in path:
            cyc = path[path.index(x):]
            w = np.array([L[a, succ[a]] for a in cyc]); a = cyc[int(w.argmin())]; succ[a] = -1
    return succ

def chains_from_succ(succ):
    n = len(succ); pred = np.full(n, -1); pred[succ[succ >= 0]] = np.where(succ >= 0)[0]
    out = []
    for st in np.where(pred < 0)[0]:
        ch = [int(st)]
        while succ[ch[-1]] >= 0: ch.append(int(succ[ch[-1]]))
        out.append(ch)
    return out

def link_metrics(succ, y, cand=None):
    n = len(succ); e = succ >= 0
    out = dict(edges=int(e.sum()), prec=float((succ[e] == np.arange(n)[e] + 1).mean()) if e.any() else 0.0,
               same=float((y[e] == y[succ[e]]).mean()) if e.any() else 0.0)
    if cand is not None:
        out["cand_rec"] = float((cand[:-1] == (np.arange(n - 1)[:, None] + 1)).any(1).mean())
    return out

DEC_OLD = dict(null_scale=1.0, lo_c=60, hi_c=160)
DEC_NEW = dict(null_scale=0.5, lo_c=80, hi_c=250)   # lead's updated fixed decoder (2026-09-23 12:00)

def decode(P, cand, lo, succ, sc, Lm, thr=-6.0, k=10, alpha=0.5, iters=5, lo_c=60, hi_c=160, p_stay=0.8, graph=None, null_scale=1.0):
    n = len(P)
    if null_scale != 1.0:
        P = P.copy(); P[:, 0] *= null_scale; P = P / P.sum(1, keepdims=True)
    g = graph if graph is not None else build_graph(cand, lo, n, k=k)
    Pg = graph_smooth(P, g, alpha=alpha, iters=iters)
    chains = chains_from_succ(cut(succ, sc, Lm, thr))
    lab, _ = calibrate_counts(Pg, chains, lo=lo_c, hi=hi_c, p_stay=p_stay)
    return lab

def sim_P(oof, st):
    n = st["n"]; P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]]; ok = ~np.isnan(P[:, 0])
    return np.where(ok[:, None], P, 1.0 / 19)

def f1(y, lab): return f1_score(y, lab, average="macro")
