"""Within-limb kNN label purity on standardised descriptors: is 'same bout' recoverable inside one limb?"""
import sys, os, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from scipy.spatial.distance import cdist
from common import LIMBS
from feats import DESC_NAMES
CACHE = r"E:\Claude code\wear\exp\hyb\agents\group\cache"
for name in sys.argv[1:]:
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    lab = z["lab"]; N = len(lab)
    # bout id = contiguous run of equal labels
    bout = np.r_[0, np.cumsum(lab[1:] != lab[:-1])]
    for l in range(4):
        X = z[f"desc{l}"].astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Dm = cdist(X, X); np.fill_diagonal(Dm, np.inf)
        for k in (5, 15):
            nn = np.argsort(Dm, axis=1)[:, :k]
            same_lab = (lab[nn] == lab[:, None]).mean()
            same_bout = (bout[nn] == bout[:, None]).mean()
            adj = (np.abs(nn - np.arange(N)[:, None]) <= 2).mean()
            print(f"{name} {LIMBS[l]:9s} k={k:2d} same-label {same_lab:.3f} same-bout {same_bout:.3f} within+-2s {adj:.3f}")
