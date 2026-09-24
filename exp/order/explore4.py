"""Can chains order bouts in time? Edge recall by transition type; set-to-set connectivity through chains."""
import os, sys, pickle
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\src")
from tlib import load_structs
from chain import cut, chains_from_succ
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
S = {}
for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
tot = {}
for s in R:
    st = S[s]; n = st["n"]; y = R[s]["y"]
    succ = cut(st["succ0"], st["sc"], st["Lm"], -6.0)
    t = np.arange(n - 1); ok = succ[t] == t + 1
    typ = np.where((y[t] == 0) & (y[t + 1] == 0), "nn", np.where((y[t] > 0) & (y[t + 1] > 0) & (y[t] == y[t + 1]), "aa", "tr"))
    for k in ("nn", "aa", "tr"):
        m = typ == k; tot.setdefault(k, [0, 0]); tot[k][0] += ok[m].sum(); tot[k][1] += m.sum()
    # |dt| distribution of chain edges by source label null/act
    e = np.where(succ >= 0)[0]; dt = np.abs(succ[e] - e)
    for k, m in (("src_null", y[e] == 0), ("src_act", y[e] > 0)):
        tot.setdefault(k, []).append(np.mean(dt[m] <= 3))
    # chain-connectivity of consecutive activity sets: chain id & position
    ch = chains_from_succ(succ); cid = np.full(n, -1); pos = np.zeros(n, int)
    for i, c in enumerate(ch): cid[c] = i; pos[c] = np.arange(len(c))
    b = np.r_[0, np.where(np.diff(y) != 0)[0] + 1, n]
    segs = [(int(y[b[i]]), b[i], b[i + 1]) for i in range(len(b) - 1)]
    acts = [(l, a0, a1) for l, a0, a1 in segs if l > 0]
    conn = 0
    for (l1, a0, a1), (l2, b0, b1) in zip(acts[:-1], acts[1:]):
        # any window of the last 5 of set1 in same chain as any window of first 5 of set2 with larger pos
        A = np.arange(max(a0, a1 - 5), a1); B = np.arange(b0, min(b1, b0 + 5))
        hit = any(cid[x] == cid[z] and pos[z] > pos[x] for x in A for z in B)
        conn += hit
    tot.setdefault("conn", [0, 0]); tot["conn"][0] += conn; tot["conn"][1] += len(acts) - 1
for k in ("nn", "aa", "tr"): print(k, "recall of true t->t+1 edge", round(tot[k][0] / tot[k][1], 3), tot[k][1])
print("chain edges |dt|<=3 frac: src_null", np.mean(tot["src_null"]).round(3), "src_act", np.mean(tot["src_act"]).round(3))
print("consecutive activity sets connected forward in the same chain:", tot["conn"])
