"""probe 1: limb encoding of w25.npz vs test meta via exact twins; exact-twin share per (subject, limb); id structure"""
import os, csv, numpy as np
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; id25 = z["id"]; s25 = z["sbj"]; l25 = z["limb"]
print("w25 keys", list(z.keys()), A25.shape, A25.dtype, "id", id25[:10], "sbj uniq", np.unique(s25), "limb uniq", np.unique(l25))
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
id26 = np.array([int(r["id"]) for r in rows]); s26 = np.array([int(r["sbj_id"]) for r in rows]); loc26 = np.array([r["sensor_location"] for r in rows])
print("A26", A26.shape, A26.dtype, "id26 == arange", np.all(id26 == np.arange(len(id26))))
diff = np.abs(A26.astype(np.float64) - A25[twin].astype(np.float64)).max((1, 2))
ex = diff < 1e-6
print("exact twins", ex.sum(), ex.mean().round(4), "| diff quantiles", np.quantile(diff, [0.5, 0.8, 0.84, 0.85, 0.9, 0.99]).round(5))
# limb encoding
for L in np.unique(loc26):
    k = ex & (loc26 == L)
    print(f"{L:10s} n26 {np.sum(loc26 == L)} exact {k.sum()} -> w25 limb codes {np.unique(l25[twin[k]], return_counts=True)}  sbj agree {np.mean(s25[twin[k]] == s26[k]):.4f}")
# crosstab 2025
print("2025 rows per (sbj, limb):")
for s in np.unique(s25):
    print(s, [int(np.sum((s25 == s) & (l25 == l))) for l in range(4)])
print("2026 rows per (sbj, loc):")
for s in np.unique(s26):
    print(s, {L: int(np.sum((s26 == s) & (loc26 == L))) for L in np.unique(loc26)})
# twin uniqueness
print("unique twins", len(np.unique(twin)), "of", len(twin), "| exact unique", len(np.unique(twin[ex])), "of", ex.sum())
# exact identical duplicates within 2025? (static tiles)
print("corr quantiles non-exact", np.quantile(corr[~ex], [0.01, 0.1, 0.5, 0.9]).round(4), "margin q", np.quantile(margin[~ex], [0.1, 0.5, 0.9]).round(4))
print("corr quantiles exact", np.quantile(corr[ex], [0.01, 0.1, 0.5]).round(4))
# id structure
print("id25 sorted?", np.all(np.diff(id25) > 0), "min/max", id25.min(), id25.max())
