"""Shared helpers for decoder experiments (sim sessions + test)."""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.metrics import f1_score
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import cut, chains_from_succ
from decode import viterbi_chains, calibrate_counts

NC = 19
WORK = r"E:\Claude code\wear\work"; PREP = r"E:\Claude code\wear\data\prep"; HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"]
EXTRA = ["sbj_2", "sbj_8", "sbj_13", "sbj_17", "sbj_4", "sbj_11"]

def blend(p1, p2, w2=0.2):
    L = (1 - w2) * np.log(np.clip(p1, 1e-6, 1)) + w2 * np.log(np.clip(p2, 1e-6, 1))
    P = np.exp(L); return P / np.nansum(P, -1, keepdims=True)

def load_oof(w2=0.2):
    lg = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy")); fu = np.load(os.path.join(WORK, "fusion_v1", "oof.npy"))
    return blend(lg, fu, w2)

def load_structs(which=("eval",)):
    out = {}
    if "eval" in which: out.update(pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")))
    if "extra" in which:
        p = os.path.join(HERE, "extra_struct.pkl")
        if os.path.exists(p): out.update(pickle.load(open(p, "rb")))
    return out

def sess_P(oof, st):
    n = st["n"]; P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]].astype(np.float64)
    ok = ~np.isnan(P[:, 0]); return np.where(ok[:, None], P, 1.0 / NC)

def mf1(y, lab):
    return f1_score(y, lab, average="macro")

# ---------------- graph (sparse re-implementation of decode.build_graph / graph_smooth) ----------------
def graph_edges(cand, lo, k=10, min_logodds=-8.0):
    """Top-k successor edges (a,b,lo) per node + top-k predecessor edges per node. Returns (rows, cols, logodds) with
    neighbour lists identical in spirit to decode.build_graph (duplicates kept)."""
    n = len(cand); valid = cand >= 0
    lo_m = np.where(valid, lo, -np.inf)
    order = np.argsort(-lo_m, 1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k); c = cand[np.arange(n)[:, None], order].ravel(); l = lo_m[np.arange(n)[:, None], order].ravel()
    ok = np.isfinite(l) & (l >= min_logodds) & (c >= 0); ea, eb, el = rows[ok], c[ok], l[ok]
    # predecessor lists: for node b, the top-k a's by lo among edges a->b
    o = np.lexsort((-el, eb)); eb_s, ea_s, el_s = eb[o], ea[o], el[o]
    start = np.r_[0, np.flatnonzero(eb_s[1:] != eb_s[:-1]) + 1]; grp_start = np.repeat(start, np.diff(np.r_[start, len(eb_s)]))
    rank = np.arange(len(eb_s)) - grp_start; kp = rank < k
    R = np.r_[ea, eb_s[kp]]; C = np.r_[eb, ea_s[kp]]; Lo = np.r_[el, el_s[kp]]
    return R, C, Lo

def graph_matrix(cand, lo, n, k=10, min_logodds=-8.0, extra=None):
    R, C, Lo = graph_edges(cand, lo, k, min_logodds); w = 1 / (1 + np.exp(-Lo))
    if extra is not None:
        R = np.r_[R, extra[0]]; C = np.r_[C, extra[1]]; w = np.r_[w, extra[2]]
    return sp.csr_matrix((w, (R, C)), shape=(n, n))

def smooth(P, W, alpha=0.5, iters=5):
    logP0 = np.log(np.clip(P, 1e-6, 1)); deg = np.asarray(W.sum(1)).ravel(); has = deg > 0
    Wn = sp.diags(1 / np.where(has, deg + 1e-9, 1)) @ W
    logP = logP0.copy()
    for _ in range(iters):
        nb = Wn @ logP
        logP = np.where(has[:, None], (1 - alpha) * logP0 + alpha * nb, logP0)
    out = np.exp(logP - logP.max(1, keepdims=True)); return out / out.sum(1, keepdims=True)

def base_chains(st, thr=-6.0):
    return chains_from_succ(cut(st["succ0"], st["sc"], np.asarray(st["Lm"], np.float32), thr))

def baseline(P, st):
    W = graph_matrix(st["cand"], st["lo"], st["n"], k=10); Pg = smooth(P, W, 0.5, 5)
    lab, _ = calibrate_counts(Pg, base_chains(st), lo=60, hi=160, p_stay=0.8); return lab
