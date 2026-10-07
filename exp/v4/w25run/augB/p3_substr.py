"""probe 3: do non-exact 2026 tiles share exact sample triples with ANY 2025 row (any limb, any offset, axis perm/sign)?
-> detects time shifts / partial overwrites / axis permutations.  Also prints a few example pairs."""
import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
print("A26 dtype", A26.dtype, "A25 dtype", A25.dtype, "A26 equal to float32 cast?", np.abs(A26 - A26.astype(np.float32)).max())
A26f = A26.astype(np.float32)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26f - A25[twin]).max((1, 2)) < 1e-6
bad = np.flatnonzero(~ex)
# value-level hash: single samples' sorted |xyz| (perm/sign invariant) -> (row, pos)
def key(v):
    return tuple(np.round(np.sort(np.abs(v)), 5))
H = {}
for i in range(len(A25)):
    for p in range(50):
        H.setdefault(key(A25[i, p]), []).append((i, p))
print("hash size", len(H))
hits = np.zeros((len(bad), 50), bool); hitrow = {}
for q, i in enumerate(bad):
    for p in range(50):
        c = H.get(key(A26f[i, p]))
        if c:
            c = [(r, pp) for r, pp in c if s25[r] == s26[i]]
            if c:
                hits[q, p] = True; hitrow.setdefault(q, []).append((p, c[:3]))
nh = hits.sum(1)
print("non-exact tiles:", len(bad), "| samples with exact (perm/sign) value hit per tile: quantiles", np.quantile(nh, [0.1, 0.5, 0.9]), "| tiles with >=25 hits", np.mean(nh >= 25).round(3), "| 0 hits", np.mean(nh == 0).round(3))
# exact tiles baseline: should be 50
# show a few examples with hits
for q in list(hitrow)[:6]:
    i = bad[q]; print(f"\n2026 row {i} sbj {s26[i]} limb {LIMBS[l26[i]]}: hits {nh[q]}")
    for p, c in hitrow[q][:6]:
        print("   pos", p, "->", [(int(r), LIMBS[l25[r]], int(pp)) for r, pp in c], "26:", A26f[i, p].round(4), "25:", A25[c[0][0], c[0][1]].round(4))
np.save(r"E:\Claude code\wear\exp\v4\w25run\augB\p3_hits.npy", hits)
