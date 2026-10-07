"""For 2026 test tiles without an identical same-limb twin: is there an identical tile in ANOTHER limb, or under an axis
permutation/sign flip, or a time shift?  (read-only probe on local data, nothing written)"""
import os, itertools, numpy as np, pandas as pd
W = r"E:\Claude code\wear"
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
tm = pd.read_csv(os.path.join(W, "data", "test", "test_meta_data.csv")); l26 = np.array([LIMBS.index(x) for x in tm.sensor_location]); s26 = tm.sbj_id.to_numpy()
diff = np.abs(A26 - A25[twin]).max((1, 2)); bad = np.flatnonzero(diff >= 1e-5)
print("non-identical", len(bad))
# hash of the first sample (rounded) of every 2025 tile, any limb
key25 = {}
for i in range(len(A25)):
    k = tuple(np.round(np.sort(np.abs(A25[i, 0])), 4))
    key25.setdefault(k, []).append(i)
hit_any, hit_other_limb, hit_perm = 0, 0, 0
for i in bad:
    k = tuple(np.round(np.sort(np.abs(A26[i, 0])), 4)); c = key25.get(k, [])
    ok = [j for j in c if np.abs(np.abs(A25[j]).sum(1) - np.abs(A26[i]).sum(1)).max() < 1e-4]
    if ok:
        hit_perm += 1
        if any(l25[j] != l26[i] for j in ok):
            hit_other_limb += 1
print(f"of {len(bad)} non-identical: identical up to axis perm/sign {hit_perm}, of which other limb {hit_other_limb}")
# time-shifted: does the 2026 tile overlap a 2025 tile shifted by k samples (same limb)?
def zs(x):
    x = x - x.mean(); return x / (np.linalg.norm(x) + 1e-9)
mag25 = np.linalg.norm(A25, axis=2); mag26 = np.linalg.norm(A26, axis=2)
print("example non-identical diff stats", np.quantile(diff[bad], [0.1, 0.5, 0.9]).round(3))
# per subject share of non-identical
print(pd.Series(np.isin(np.arange(len(A26)), bad)).groupby(s26).mean().round(3))
