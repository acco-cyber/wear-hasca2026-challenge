"""how much can a boundary refiner fix at all: oracle flips on the refiner's candidate (tile, other) pairs"""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1
import rlib as R

d = R.load_cache(); y, fold, fin = d["y"], d["fold"], d["fin_o"]
z = np.load(os.path.join(R.HERE, "preds", "base.npz")); tl, ot, pr, T, K = z["tl"], z["ot"], z["pr"], z["T"], int(z["K"])
err = fin != y
print(f"errors {err.sum()} of {len(y)} ({err.mean():.4f}); F1 {macro_f1(y, fin):.4f}")
key = tl * 32 + ot; uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
g, o = uk // 32, uk % 32; good = y[g] == o
print(f"unique candidate pairs {len(uk)}, of which correct flips {good.sum()} (tiles {len(np.unique(g[good]))})")
orc = fin.copy(); orc[g[good]] = o[good]
print(f"oracle on candidates: F1 {macro_f1(y, orc):.4f}; errors left {np.sum(orc != y)}")
s = np.bincount(inv, pr, len(uk)) / K; mx = np.maximum.reduceat(pr[np.argsort(inv, kind='stable')], np.r_[0, np.cumsum(cnt)[:-1]])
for thr in (0.3, 0.4, 0.5, 0.6):
    sel = s > thr
    print(f"thr {thr}: pairs {sel.sum()}, correct {np.sum(sel & good)}, wrong {np.sum(sel & ~good)}")
# where are the remaining errors: distance (in true chain) from a decoded boundary
ts = d["true_succ"]; n = len(y)
prv = np.full(n, -1); prv[ts[ts >= 0]] = np.flatnonzero(ts >= 0)
cand_tiles = np.zeros(n, bool); cand_tiles[g] = True
print(f"errors covered by any candidate row: {np.mean(cand_tiles[err]):.3f}")
# error tiles: fraction where true label == decoded label of a true neighbour
nb_ok = np.zeros(n, bool)
for a in (ts, prv):
    m = a >= 0; nb_ok[m] |= fin[a[m]] == y[m]
print(f"errors whose TRUE neighbour has the error tile's true label as decoded: {np.mean(nb_ok[err]):.3f}")
# class confusion of errors
from collections import Counter
print("top confusions (true, pred):", Counter(zip(y[err].tolist(), fin[err].tolist())).most_common(12))
