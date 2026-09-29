"""ceilings WITHOUT self vote (self only breaks ties): label of window = majority TRUE label of its neighbours"""
import numpy as np
from common import *
d = load_oof(); y, rec, st, sbj = d["y"], d["rec"], d["start"], d["sbj"]; n = len(y); succ = d["succ"]
bid = true_bouts(y, rec, st)


def maj(NB, valid):
    votes = np.zeros((n, N_CLS)); rr = np.repeat(np.arange(n), NB.shape[1])[valid.ravel()]
    np.add.at(votes, (rr, y[NB.ravel()[valid.ravel()]]), 1.0); votes[np.arange(n), y] += 0.01
    return macro_f1(y, votes.argmax(1))


for nm in ("raw_p0.0", "centred_p0.0", "raw_p0.5", "centred_p0.5", "raw+link0.5_p0.0", "raw+link0.5_p0.5"):
    nb = np.load(f"nb_{nm}.npy")
    print(nm, " | ".join(f"k={k}: {maj(nb[:, :k], np.ones((n, k), bool)):.4f} (same-bout {np.mean(bid[nb[:, :k]] == bid[:, None]):.3f})" for k in (5, 10, 20)))
ms = succ >= 0; prv = np.full(n, -1); prv[succ[ms]] = np.flatnonzero(ms)
for H in (1, 2, 3):
    nbrs = []; fw = np.arange(n); bw = np.arange(n)
    for h in range(H):
        fw = np.where(fw >= 0, succ[np.maximum(fw, 0)], -1); bw = np.where(bw >= 0, prv[np.maximum(bw, 0)], -1); nbrs += [fw.copy(), bw.copy()]
    NB = np.stack(nbrs, 1); v = NB >= 0
    print(f"link hops {H}: {maj(np.maximum(NB, 0), v):.4f} (same-bout {np.mean((bid[np.maximum(NB, 0)] == bid[:, None])[v]):.3f})")
