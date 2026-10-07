"""Gate A diagnostic: what the chain-rescored links fix. Per link set: linked share, exact successor, cross-label rate
(linked rows whose successor carries another label; the quantity that drove OOF in research2/headroom/links_typed.log),
and exact rate on true label-boundary rows.  python a4_linkstats.py name=links.npz [...]"""
import os, sys, numpy as np
W = r"E:\Claude code\wear"
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
y = st["oof_y"].astype(np.int64); ts = st["true_succ"].astype(np.int64); fold = st["oof_fold"].astype(np.int64); sens = st["sensor_oof"].astype(np.int64)
h = ts >= 0; bnd = h & (y != y[np.maximum(ts, 0)])
sets = [("K7own", os.path.join(K7, "links.npz"))] + [tuple(a.split("=", 1)) for a in sys.argv[1:]]
print(f"rows {len(y)}, with true successor {h.sum()}, true label boundaries {bnd.sum()} (true cross-label rate {bnd.sum() / h.sum():.4f})")
for nm, p in sets:
    z = np.load(p); S = z["oof_succ"].astype(np.int64)
    r = []
    for k in range(len(S)):
        s = S[k]; m = s >= 0
        cl = m & (y != y[np.maximum(s, 0)])
        r.append([m.mean(), (s[h] == ts[h]).mean(), cl.sum() / m.sum(), (cl & ((y == 0) | (y[np.maximum(s, 0)] == 0))).sum() / m.sum(),
                  (s[bnd] == ts[bnd]).mean(), (cl & (s != ts)).sum() / m.sum()])
    r = np.array(r)
    s0 = S[0]; m0 = s0 >= 0; same = h & (sens == sens[np.maximum(ts, 0)])
    print(f"{nm:>10}: member0 linked {r[0, 0]:.4f} exact {r[0, 1]:.4f} cross-label {r[0, 2]:.4f} (null-inv {r[0, 3]:.4f}, wrong&cross {r[0, 5]:.4f}) "
          f"boundary-exact {r[0, 4]:.4f} | mean8 exact {r[:, 1].mean():.4f} cross-label {r[:, 2].mean():.4f} wrong&cross {r[:, 5].mean():.4f} | "
          f"exact same-limb {(s0[same] == ts[same]).mean():.4f} cross-limb {(s0[h & ~same] == ts[h & ~same]).mean():.4f}", flush=True)
