import sys, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import *

for name in sys.argv[1:]:
    acc, lab = load_session(name)
    nsec = acc.shape[1]
    mag = np.sqrt((np.nan_to_num(acc) ** 2).sum(-1)).reshape(4, -1)
    L = 100; B = 30 * 50
    print("====", name, nsec, "s")
    for (a, b) in [(1, 3), (0, 2), (0, 1), (2, 3)]:
        rows = []
        for s0 in range(L, mag.shape[1] - B - L, B):
            x = mag[a, s0:s0 + B]; x = x - x.mean()
            if x.std() < 0.1: continue
            cc = np.array([np.corrcoef(x, mag[b, s0 + l:s0 + B + l])[0, 1] for l in range(-L, L + 1)])
            k = np.argmax(cc)
            labs = lab[s0 // 50:(s0 + B) // 50]
            v, c = np.unique(labs, return_counts=True)
            rows.append((s0 // 50, k - L, round(float(cc[k]), 2), round(float(cc[L]), 2), v[np.argmax(c)][:18]))
        print(LIMBS[a], LIMBS[b])
        for r in rows:
            if r[2] > 0.35:
                print("   t=%5d best_lag=%4d peak=%.2f lag0=%.2f %s" % r)
