"""probe 2: for non-exact 2026 tiles, where is the true counterpart in 2025?  search ALL limbs of the same subject by
(a) magnitude distance (rotation/sign invariant), (b) raw distance, (c) sorted |axis| distance (perm+sign invariant)"""
import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
bad = np.flatnonzero(~ex)
M25 = np.linalg.norm(A25, axis=2); M26 = np.linalg.norm(A26, axis=2)
P25 = np.sort(np.abs(A25), axis=2).reshape(len(A25), -1); P26 = np.sort(np.abs(A26), axis=2).reshape(len(A26), -1)
R25 = A25.reshape(len(A25), -1); R26 = A26.reshape(len(A26), -1)


def nn(Q, D):
    d = (Q ** 2).sum(1)[:, None] + (D ** 2).sum(1)[None] - 2 * Q @ D.T
    j = d.argmin(1); return j, np.sqrt(np.maximum(d[np.arange(len(Q)), j], 0))


res = {}
for name, X25, X26 in (("mag", M25, M26), ("sortabs", P25, P26), ("raw", R25, R26)):
    J = np.full(len(bad), -1); D = np.zeros(len(bad))
    for s in np.unique(s26):
        q = bad[s26[bad] == s]; c = np.flatnonzero(s25 == s)
        j, d = nn(X26[q], X25[c]); J[s26[bad] == s] = c[j]; D[s26[bad] == s] = d
    res[name] = (J, D)
    scale = np.sqrt((X26[bad] ** 2).sum(1))
    rel = D / (scale + 1e-9)
    print(f"[{name}] rel dist quantiles {np.quantile(rel, [0.1, 0.25, 0.5, 0.75, 0.9]).round(4)} | <1e-4: {np.mean(rel < 1e-4):.3f}")
    same = l25[J] == l26[bad]
    print(f"   nn limb == own limb {same.mean():.3f}; nn limb distribution by own limb:")
    for L in (0, 2):
        k = l26[bad] == L
        print(f"   own {LIMBS[L]} n={k.sum()}: ", {LIMBS[x]: int(np.sum(l25[J[k]] == x)) for x in range(4)})
np.savez(r"E:\Claude code\wear\exp\v4\w25run\augB\p2_nn.npz", bad=bad, **{f"J_{k}": v[0] for k, v in res.items()}, **{f"D_{k}": v[1] for k, v in res.items()})
