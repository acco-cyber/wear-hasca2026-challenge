"""diagnostic: is assembly agreement informative about base-link correctness beyond the link's own score?"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
W = r"E:\Claude code\wear"
TD = os.path.join(W, "exp", "v4", "w25run", "testD"); K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
S = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
ts = S["true_succ"].astype(np.int64); y = S["oof_y"]; N = len(ts)
LK = np.load(os.path.join(TD, "links_oof_K7_deaug.npz")); su = LK["oof_succ"][0].astype(np.int64); sc = LK["oof_score"][0]
G = np.load(os.path.join(HERE, "cache", "sim_grid1.npz"))
h = (ts >= 0) & (su >= 0); ok = su == ts; cross = np.zeros(N, bool); cross[su >= 0] = (y[su >= 0] != y[su[su >= 0]]) & ~ok[su >= 0]
dec = np.digitize(sc, np.quantile(sc[h], np.linspace(0, 1, 11)[1:-1]))
for c in sys.argv[1].split(","):
    asu, root, coord, gt = G[c + "_asu"], G[c + "_root"], G[c + "_coord"], G[c + "_gtiles"]
    for gmin in (2, 10):
        use = (asu >= 0) & (gt >= gmin)
        hp = np.full(N, -1); hp[asu[use]] = np.flatnonzero(use)
        j = np.maximum(su, 0)
        agree = use & (asu == su)
        contra = (su >= 0) & ~agree & ((use & (asu != su)) | (hp[j] >= 0) & (hp[j] != np.arange(N)) | ((gt >= gmin) & (root[j] == root) & (coord[j] != coord + 1)))
        none = (su >= 0) & ~agree & ~contra
        print(f"{c} gmin {gmin}: share agree {agree[h].mean():.3f} contra {contra[h].mean():.3f} none {none[h].mean():.3f} | prec agree {ok[h & agree].mean():.4f} contra {ok[h & contra].mean():.4f} none {ok[h & none].mean():.4f}"
              f" | wrong&cross share agree {cross[h & agree].mean():.4f} contra {cross[h & contra].mean():.4f} none {cross[h & none].mean():.4f}")
        row = []
        for d in range(10):
            m = h & (dec == d)
            row.append(f"d{d}: A {ok[m & agree].mean() if (m & agree).any() else -1:.3f}({(m & agree).sum()}) C {ok[m & contra].mean() if (m & contra).any() else -1:.3f}({(m & contra).sum()}) N {ok[m & none].mean() if (m & none).any() else -1:.3f}({(m & none).sum()})")
        print("   by score decile: " + " | ".join(row))
