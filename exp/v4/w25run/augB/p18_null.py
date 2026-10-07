"""probe 18: null spaces of the warp matrices, restore error by position per kind, matched-filter R0 statistic"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=3)
W = r"E:\Claude code\wear"; D = os.path.dirname(os.path.abspath(__file__))
import aug
for nm, Wm, tau in (("WL", aug.WL, aug.TAU_L), ("WR", aug.WR, aug.TAU_R)):
    U, S, Vt = np.linalg.svd(Wm)
    print(nm, "smallest singular values", S[-6:].round(6))
    for v in Vt[S < 1e-9 * S[0]]:
        sup = np.flatnonzero(np.abs(v) > 1e-6); print("  null vector support", sup, "values", v[sup].round(3))
    used = np.zeros(50, int)
    for t in tau:
        k = min(int(np.floor(t)), 48); used[k] += 1; used[k + 1] += 1
    print("  interval hits (k -> number of b samples whose bracket starts at k)", np.bincount(np.minimum(np.floor(tau).astype(int), 48), minlength=49))
v = np.load(os.path.join(D, "validate.npz")); twin25, kind_t = v["twin25"], v["kind_t"]
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; l25 = z["limb"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float32)
for k in (0, 2):
    q = np.flatnonzero(kind_t == k)
    e = np.abs(aug.restore(A25[twin25[q]], np.full(len(q), k)) - A26[q])
    print(aug.KINDS[k], "restore rms by position\n", np.sqrt((e ** 2).mean((0, 2))).round(4))
# matched filter statistics for R0 / R1 using difference order d
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ra = np.flatnonzero(l26 == 2)
B = A25[twin25[ra]].astype(np.float64); kt = kind_t[ra]
for d in (1, 2, 3, 4):
    for nm, N in (("N0", aug.N0), ("N1", aug.N1)):
        HN = np.diff(N, d, axis=0); HB = np.diff(B, d, axis=1)
        t = (HB * HN).sum((1, 2)) / (HN ** 2).sum()
        tgt = 1 if nm == "N0" else 2
        pos = t[kt == tgt]; neg = t[kt != tgt]
        thr = 0.5
        print(f"d={d} {nm}: stat on true kind q01/q50 {np.quantile(pos, [0.001, 0.01, 0.5]).round(3)} | others q50/q99/q999/max {np.quantile(neg, [0.5, 0.99, 0.999]).round(3)} {neg.max():.3f} | errors at 0.5: miss {np.sum(pos < thr)} fp {np.sum(neg > thr)}")
