"""diagnostic only (uses true positions for SCORING): chain fragment statistics in the deaug simulation"""
import os, sys
import numpy as np
TD = r"E:\Claude code\wear\exp\v4\w25run\testD"
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
S = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
sbj = S["oof_sbj"]; sens = S["sensor_oof"]; ts = S["true_succ"]; N = len(sbj)
for s in [int(x) for x in sys.argv[1].split(",")]:
    z = np.load(os.path.join(TD, "cache", f"chaindeaug_s{s}.npz")); n = len(z["rows"])
    loc = np.flatnonzero(sbj == s); p_of = np.full(N, -1); p_of[z["rows"]] = np.arange(n)
    # true next position exists? (recording boundary): position p -> p+1 is the true successor iff ts[rows[p]] == rows[p+1]
    tnext = np.full(n, -1); r = z["rows"]; ok = ts[r] >= 0; tnext[ok] = p_of[ts[r][ok]]
    for L in range(4):
        su = z[f"succ{L}"]; cf = z[f"conf{L}"]
        corr = su == tnext
        line = [f"sbj {s} L{L} aug {z['aug'][L].mean():.3f} exact {np.mean(corr):.3f}"]
        for thr in (0.5, 0.8, 0.9):
            m = (su >= 0) & (cf >= thr)
            nxt = np.where(m, su, -1); hp = np.zeros(n, bool); hp[nxt[nxt >= 0]] = True
            lens = []
            for hd in np.flatnonzero(~hp):
                q = hd; o = 0
                while q >= 0:
                    o += 1; q = nxt[q]
                lens.append(o)
            lens = np.array(lens)
            line.append(f"thr {thr}: links {m.mean():.3f} prec {corr[m].mean():.3f} frags {len(lens)} meanlen {lens.mean():.1f} nodes-in-len>=10 {lens[lens >= 10].sum() / n:.3f} >=30 {lens[lens >= 30].sum() / n:.3f}")
        print(" | ".join(line))
