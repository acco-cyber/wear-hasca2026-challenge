"""Temporal-neighbourhood test through the link graph: random walk from a true bout; mass reaching other activities.
Does the walk from the 11-bout reach 13 more than 14 (and 12-bout reach 14 more than 13)? Are true temporal neighbours
of a bout over-represented in its walk profile?"""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, scipy.sparse as sp
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\exp\decoder")
from tlib import load_structs
from common import graph_matrix
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
S = {}
for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
FAM = np.zeros(19, int); FAM[1:6] = 1; FAM[6:11] = 2; FAM[11:13] = 3; FAM[13:15] = 4; FAM[15] = 5; FAM[16:18] = 6; FAM[18] = 7
res = {t: [] for t in (5, 20, 50)}; pair = {t: [0, 0] for t in (5, 20, 50)}
for s in R:
    st = S[s]; n = st["n"]; y = R[s]["y"]
    W = graph_matrix(st["cand"], st["lo"], n, k=10); W = (W + W.T).tocsr(); d = np.asarray(W.sum(1)).ravel(); M = sp.diags(1 / np.maximum(d, 1e-9)) @ W
    # true temporal neighbours of each activity: previous and next distinct activity in time order
    acts = [a for a in y if a > 0]; comp = [a for i, a in enumerate(acts) if i == 0 or a != acts[i - 1]]
    nbr = {c: set() for c in range(1, 19)}
    for a, b in zip(comp[:-1], comp[1:]): nbr[a].add(b); nbr[b].add(a)
    prof = {}
    for c in range(1, 19):
        G = np.where(y == c)[0]
        if len(G) == 0: continue
        pi = np.zeros(n); pi[G] = 1 / len(G); out = {}
        for t in range(1, 51):
            pi = M.T @ pi
            if t in res:
                mass = np.bincount(y, weights=pi, minlength=19)
                other = np.array([k for k in range(1, 19) if FAM[k] != FAM[c]])
                mo = mass[other] / mass[other].sum()
                isn = np.array([k in nbr[c] for k in other])
                if isn.any() and (~isn).any(): res[t].append(mo[isn].mean() / mo[~isn].mean())
                out[t] = mass
        prof[c] = out
    for t in pair:
        if all(c in prof for c in (11, 12, 13, 14)):
            m11, m12 = prof[11][t], prof[12][t]
            pair[t][0] += (m11[13] / m11[14] > m12[13] / m12[14]); pair[t][1] += 1
for t in res: print(f"t={t}: mean ratio (walk mass per true-temporal-neighbour activity / per non-neighbour) {np.mean(res[t]):.2f} median {np.median(res[t]):.2f} | 11->13 & 12->14 pairing {pair[t]}")
