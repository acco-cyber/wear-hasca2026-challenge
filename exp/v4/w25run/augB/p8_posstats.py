"""probe 8: positional statistics of 2025 rows by group (claimed-exact = clean, unclaimed) per limb:
duplicate-sample artifacts, per-position diff energy, mean-centered positional pattern (fixed-noise test)"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=4)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
claimed = np.zeros(len(A25), bool); claimed[twin[ex]] = True
dup01 = np.all(A25[:, 0] == A25[:, 1], 1); dup4849 = np.all(A25[:, 48] == A25[:, 49], 1)
anydup = np.array([np.any(np.all(A25[i, 1:] == A25[i, :-1], 1)) for i in range(len(A25))])
print("2025 rows with b0==b1:")
for L in range(4):
    for s in np.unique(s25):
        k = (l25 == L) & (s25 == s)
        print(f"  {LIMBS[L]:9s} sbj {s}: n {k.sum()} dup01 {dup01[k].sum()} ({dup01[k].mean():.3f}) dup4849 {dup4849[k].sum()} anydup {anydup[k].sum()} | claimed {claimed[k].sum()} claimed&dup01 {(claimed & dup01)[k].sum()}")
# same for 2026 tiles
d26 = np.all(A26[:, 0] == A26[:, 1], 1); print("2026 tiles b0==b1:", d26.sum())
# positional diff energy
for L in range(4):
    for nm, k in (("claimed", claimed & (l25 == L)), ("unclaimed", ~claimed & (l25 == L)), ("uncl&!dup01", ~claimed & ~dup01 & (l25 == L))):
        X = A25[k]; de = (np.diff(X, axis=1) ** 2).sum(2)
        pr = de.mean(0) / de.mean()
        cen = (X - X.mean(1, keepdims=True)).mean(0)  # fixed additive pattern would survive averaging
        print(f"{LIMBS[L]:9s} {nm:12s} n {k.sum():5d} diffE rel pos[0:6] {pr[:6]} mid {pr[20:24]} end {pr[-4:]} | mean centered pattern norm {np.linalg.norm(cen):.4f}")
