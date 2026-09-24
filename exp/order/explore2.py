"""Chain statistics on sim sessions: chain lengths, how often a chain spans >=2 true activities, and order info."""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive"); sys.path.insert(0, r"E:\Claude code\wear\src")
from tlib import load_structs
from chain import cut, chains_from_succ
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
S = {}
for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
FAM = np.zeros(19, int); FAM[1:6] = 1; FAM[6:11] = 2; FAM[11:13] = 3; FAM[13:15] = 4; FAM[15] = 5; FAM[16:18] = 6; FAM[18] = 7
for s in R:
    st = S[s]; n = st["n"]; y = R[s]["y"]; lab = R[s]["lab"]
    ch = chains_from_succ(cut(st["succ0"], st["sc"], st["Lm"], -6.0))
    L = np.array([len(c) for c in ch])
    # true temporal succ precision
    succ = cut(st["succ0"], st["sc"], st["Lm"], -6.0); e = succ >= 0
    prec = np.mean(succ[e] == np.arange(n)[e] + 1)
    multi = 0; trans_ok = 0; trans_tot = 0; chain_idx_span = []
    for c in ch:
        acts = [a for a in y[c] if a > 0]
        comp = [a for i, a in enumerate(acts) if i == 0 or a != acts[i - 1]]
        if len(set(comp)) >= 2: multi += 1
        chain_idx_span.append(max(c) - min(c))
        # consecutive (in chain) distinct-activity transitions: are they true temporal order?
    print(f"{s}: n {n} chains {len(ch)} len med {np.median(L):.0f} mean {L.mean():.1f} max {L.max()} | frac windows in chains>=20: {L[L>=20].sum()/n:.2f} | "
          f"succ prec {prec:.2f} | chains w/ >=2 true acts {multi} | median time-span {np.median(chain_idx_span):.0f}")
