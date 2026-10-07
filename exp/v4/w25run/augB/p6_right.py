"""probe 6: right_arm non-exact tiles: search unclaimed 2025 right_arm rows of the same subject for the best linear fit
b ~ X @ M (X = a or warp(a), optionally + offset).  Reports residual of best and margin to second best."""
import os, csv, numpy as np
np.set_printoptions(linewidth=220, suppress=True)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
tau = np.median(np.load(os.path.join(D, "p5_pairs.npz"))["tau"], 0)
Wop = np.zeros((50, 50))
for p, t in enumerate(tau):
    k = min(int(np.floor(t)), 48); f = t - k; Wop[p, k] += 1 - f; Wop[p, k + 1] += f
claimed = np.zeros(len(A25), bool); claimed[twin[ex]] = True
L = int(os.environ.get("LIMB", "2"))
bad = np.flatnonzero(~ex & (l26 == L))
print(LIMBS[L], "non-exact", len(bad))
res = {}
for name, warp, off in (("lin", False, False), ("lin+off", False, True), ("warp lin", True, False), ("warp lin+off", True, True)):
    best = np.zeros(len(bad), np.int64); r1 = np.zeros(len(bad)); r2 = np.zeros(len(bad)); Ms = []
    for s in np.unique(s26[bad]):
        q = bad[s26[bad] == s]; c = np.flatnonzero((s25 == s) & (l25 == L) & ~claimed)
        B = A25[c]; Bn = (B ** 2).sum((1, 2))
        for qi, i in zip(np.flatnonzero(s26[bad] == s), q):
            X = Wop @ A26[i] if warp else A26[i]
            if off:
                X = np.c_[X, np.ones(50)]
            Q, _ = np.linalg.qr(X)
            proj = np.einsum("pk,npj->nkj", Q, B)
            rr = (Bn - (proj ** 2).sum((1, 2))) / Bn
            o = np.argsort(rr)[:2]; best[qi] = c[o[0]]; r1[qi] = rr[o[0]]; r2[qi] = rr[o[1]]
    res[name] = (best, r1, r2)
    print(f"[{name:13s}] best rel resid q {np.quantile(r1, [0.1, 0.25, 0.5, 0.75, 0.9]).round(5)} | second q {np.quantile(r2, [0.1, 0.5, 0.9]).round(4)} | ratio r1/r2 <0.1: {np.mean(r1 < 0.1 * r2):.3f}")
np.savez(os.path.join(D, f"p6_L{L}.npz"), bad=bad, **{f"best_{k.replace(' ', '_').replace('+', 'p')}": v[0] for k, v in res.items()},
         **{f"r1_{k.replace(' ', '_').replace('+', 'p')}": v[1] for k, v in res.items()}, **{f"r2_{k.replace(' ', '_').replace('+', 'p')}": v[2] for k, v in res.items()})
