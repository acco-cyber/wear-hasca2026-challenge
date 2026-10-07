"""probe 19: inspect matched-filter false positives (clean-labelled right-arm twin rows with high N0 statistic)"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=4)
W = r"E:\Claude code\wear"; D = os.path.dirname(os.path.abspath(__file__))
import aug
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
v = np.load(os.path.join(D, "validate.npz")); twin25, kind_t = v["twin25"], v["kind_t"]
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); l25 = z["limb"]; s25 = z["sbj"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])


def stat(B, N, d=3):
    HN = np.diff(N, d, axis=0); HB = np.diff(B, d, axis=1); return (HB * HN).sum((1, 2)) / (HN ** 2).sum()


t0 = stat(A26, aug.N0); t1 = stat(A26, aug.N1)
print("2026 tiles (all clean by construction?) N0 stat >0.5:", np.flatnonzero(t0 > 0.5), "N1 stat > 0.5:", np.flatnonzero(t1 > 0.5))
for i in np.flatnonzero((t0 > 0.5) | (t1 > 0.5))[:6]:
    print(f" tile {i} limb {LIMBS[l26[i]]} kind_t {kind_t[i]} t0 {t0[i]:.3f} t1 {t1[i]:.3f}")
    a = A26[i]; print("  tile[:6]\n", a[:6]); print("  tile - N0 [:6]\n", (a - aug.N0)[:6])
    # restored version: if this 2026 tile is itself R0-augmented, its de-augmented form should be smooth
    r = (a - aug.N0) @ aug.M0.T
    print("  rough tile", (np.diff(a, 2, axis=0) ** 2).sum().round(4), "rough tile-N0", (np.diff(a - aug.N0, 2, axis=0) ** 2).sum().round(4))
# distribution on all 2025 rows (all limbs) - legs are clean
for L in range(4):
    j = l25 == L; s0 = stat(A25[j], aug.N0); s1 = stat(A25[j], aug.N1)
    print(f"2025 {LIMBS[L]:9s}: N0 stat hist", np.histogram(s0, bins=[-9, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 9])[0], "N1 stat hist", np.histogram(s1, bins=[-9, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 9])[0])
