"""probe 11: find right_arm pairs under the additive-pattern hypothesis b = a + N_k (+const), refine N_k from pairs and
check exactness (spread of b - a across pairs)"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=5)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
claimed = np.zeros(len(A25), bool); claimed[twin[ex]] = True
pz = np.load(os.path.join(D, "p10_rpat.npz")); pats = {1: pz["A"], 2: pz["B"]}
L = 2
bad = np.flatnonzero(~ex & (l26 == L))
Ac = A26 - A26.mean(1, keepdims=True); Bc = A25 - A25.mean(1, keepdims=True)
out = np.full((len(bad), 3), np.nan); arg = np.full((len(bad), 3), -1)
for s in np.unique(s26[bad]):
    qq = np.flatnonzero(s26[bad] == s); c = np.flatnonzero((s25 == s) & (l25 == L) & ~claimed)
    for k in (0, 1, 2):
        Npat = 0 if k == 0 else pats[k]
        T = (Bc[c] - Npat).reshape(len(c), -1)                         # candidate minus pattern
        Q = Ac[bad[qq]].reshape(len(qq), -1)
        d = (Q ** 2).sum(1)[:, None] + (T ** 2).sum(1)[None] - 2 * Q @ T.T
        j = d.argmin(1); out[qq, k] = np.sqrt(np.maximum(d[np.arange(len(qq)), j], 0)); arg[qq, k] = c[j]
kbest = out.argmin(1)
print("best pattern counts (0 none,1 A,2 B):", np.bincount(kbest, minlength=3))
print("best dist quantiles", np.quantile(out.min(1), [0.1, 0.5, 0.9]).round(4))
# refine N_k from pairs: D = b - a (uncentered)
for k in (1, 2):
    sel = np.flatnonzero(kbest == k)
    Dk = A25[arg[sel, k]] - A26[bad[sel]]
    med = np.median(Dk, 0); dev = np.abs(Dk - med).max((1, 2))
    print(f"pattern {k}: pairs {len(sel)} | max|b-a-median| quantiles {np.quantile(dev, [0.1, 0.25, 0.5, 0.75, 0.9]).round(5)}")
    print("  median D[:5]\n", med[:5])
    # also check: b - a constant offset varies? subtract per-pair mean
    Dkc = Dk - Dk.mean(1, keepdims=True); medc = np.median(Dkc, 0); devc = np.abs(Dkc - medc).max((1, 2))
    print(f"  centered: dev quantiles {np.quantile(devc, [0.1, 0.25, 0.5, 0.75, 0.9]).round(5)}")
np.savez(os.path.join(D, "p11_rpairs.npz"), bad=bad, out=out, arg=arg, kbest=kbest)
