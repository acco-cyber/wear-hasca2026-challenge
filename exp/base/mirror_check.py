import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
imu = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r")
m = meta(); y = m.y.to_numpy(); ok = m.pur.to_numpy() >= 0.8
g = np.nanmean(np.asarray(imu[:, :, :, :], np.float32), 2)   # (N,4,3) mean per window
sd = np.nanstd(np.asarray(imu[:, :, :, :], np.float32), 2)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for c in [0, 1, 6, 11, 13, 15, 16, 18]:
    mm = ok & (y == c)
    print("class", c, "mean g per limb [LA, LL, RA, RL]:\n", np.nanmean(g[mm], 0), "\n  sd:", np.nanmean(sd[mm], 0))
