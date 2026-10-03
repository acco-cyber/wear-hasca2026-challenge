"""How many true same-second pairs survive a cheap |d logjerk|,|d logstd| pre-filter vs how many random pairs?"""
import sys, os, glob, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import PAIRS, LIMBS
from feats import DESC_NAMES
CACHE = r"E:\Claude code\wear\exp\hyb\agents\group\cache"
iJ = DESC_NAMES.index("logjerk"); iS = DESC_NAMES.index("logstd")
rng = np.random.default_rng(0)
dj_true = {p: [] for p in PAIRS}; ds_true = {p: [] for p in PAIRS}; dj_rand = {p: [] for p in PAIRS}; ds_rand = {p: [] for p in PAIRS}
for f in sorted(glob.glob(os.path.join(CACHE, "sbj_*.npz"))):
    z = np.load(f, allow_pickle=True)
    for (a, b) in PAIRS:
        A = z[f"desc{a}"]; B = z[f"desc{b}"]; n = len(A)
        j = rng.integers(0, n, n)
        dj_true[(a, b)].append(np.abs(A[:, iJ] - B[:, iJ])); ds_true[(a, b)].append(np.abs(A[:, iS] - B[:, iS]))
        dj_rand[(a, b)].append(np.abs(A[:, iJ] - B[j, iJ])); ds_rand[(a, b)].append(np.abs(A[:, iS] - B[j, iS]))
for (a, b) in PAIRS:
    djt = np.concatenate(dj_true[(a, b)]); dst = np.concatenate(ds_true[(a, b)])
    djr = np.concatenate(dj_rand[(a, b)]); dsr = np.concatenate(ds_rand[(a, b)])
    print(f"{LIMBS[a]}-{LIMBS[b]}: true |dlogjerk| q50/q95/q99 {np.quantile(djt,[.5,.95,.99]).round(2)}  true |dlogstd| q95/q99 {np.quantile(dst,[.95,.99]).round(2)}")
    for tj in (1.0, 1.5, 2.0):
        for ts in (1.0, 1.5, 2.0):
            keep_t = ((djt < tj) & (dst < ts)).mean(); keep_r = ((djr < tj) & (dsr < ts)).mean()
            print(f"    tj={tj} ts={ts}: keeps true {keep_t:.4f} random {keep_r:.3f}")
