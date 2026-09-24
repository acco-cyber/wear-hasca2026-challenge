"""mrf4 decoder re-implementation with a structured chain transition matrix (logT) and a structured ICM coupling
matrix M (Potts = identity). With logT=None, M=None it reproduces decoder.decode_subject(P, st, VARIANTS['mrf4'])."""
import os, sys
import numpy as np, scipy.sparse as sp
DEC = r"E:\Claude code\wear\exp\decoder"
if DEC not in sys.path: sys.path.insert(0, DEC)
from common import graph_matrix, smooth, NC
from nbvit import pack, viterbi, logT_matrix
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import cut, chains_from_succ

def prep(st, n):
    W = graph_matrix(st["cand"], st["lo"], n, k=10)
    chains = chains_from_succ(cut(st["succ0"], st["sc"], np.asarray(st["Lm"], np.float32), -6.0)); pk = pack(chains)
    Wm = W + W.T; d = np.asarray(Wm.sum(1)).ravel(); Wm = (sp.diags(1 / np.where(d > 0, d, 1)) @ Wm).tocsr()
    return dict(W=W, pk=pk, Wm=Wm)

def icm(logE, W, lab, lam, iters, M=None, frac=0.5):
    rng = np.random.RandomState(0); n = len(lab)
    for _ in range(iters):
        O = np.zeros((n, NC)); O[np.arange(n), lab] = 1
        V = W @ O
        if M is not None: V = V @ M
        new = (logE + lam * V).argmax(1); upd = rng.rand(n) < frac; lab = np.where(upd, new, lab)
    return lab

def decode(P, st, pre=None, logT=None, M=None, lo=80, hi=250, ps=0.7, lam=4.0, icm_iters=10, ns=0.5, iters=40, step=0.25, b_fixed=None):
    n = len(P); pre = pre or prep(st, n)
    P = P.copy(); P[:, 0] *= ns; P = P / P.sum(1, keepdims=True)
    logP = np.log(np.clip(smooth(P, pre["W"], 0.5, 5), 1e-6, 1))
    logT = logT_matrix(ps) if logT is None else logT
    b = np.zeros(NC); lo_a = np.broadcast_to(np.asarray(lo, float), (NC - 1,)); hi_a = np.broadcast_to(np.asarray(hi, float), (NC - 1,))
    for it in range(iters):
        Z = logP + b[None]; Z = np.maximum(Z - np.logaddexp.reduce(Z, 1, keepdims=True), np.log(1e-6))
        lab = viterbi(Z, pre["pk"], logT)
        if lam > 0: lab = icm(Z, pre["Wm"], lab, lam, icm_iters, M)
        cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo_a)[0] + 1; over = np.where(cnt[1:] > hi_a)[0] + 1
        if len(under) == 0 and len(over) == 0: break
        b[under] += step; b[over] -= step
    return lab
