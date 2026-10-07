"""probe 9: right_arm fixed additive pattern.  Estimate pattern from unclaimed-vs-claimed positional means, score every
2025 right_arm row by projection, look at the score distribution; re-estimate N from flagged rows; then search pairs
b - N ~ a @ M + c for non-exact 2026 right_arm tiles"""
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
L = 2
R = np.flatnonzero(l25 == L)
X = A25[R]; Xc = X - X.mean(1, keepdims=True)
P = Xc[~claimed[R]].mean(0) - Xc[claimed[R]].mean(0)
print("pattern norm", np.linalg.norm(P).round(4)); print("pattern (first 10 samples)\n", P[:10])
sc = (Xc * P).sum((1, 2)) / (P ** 2).sum()
h, e = np.histogram(sc, bins=np.linspace(-1, 4, 26)); print("projection score hist (claimed+unclaimed)"); print(np.c_[e[:-1].round(2), h])
hc, _ = np.histogram(sc[claimed[R]], bins=e); print("claimed only", hc)
flag = sc > 1.2 * np.median(sc[sc > 0.5 * sc.max()]) * 0 + 0.5 * (np.median(sc[claimed[R]]) + np.quantile(sc, 0.8))
# better: iterate N estimate
thr = 0.5 * (np.median(sc[claimed[R]]) + np.quantile(sc, 0.9))
for it in range(5):
    fl = sc > thr
    N = Xc[fl].mean(0) - Xc[claimed[R]].mean(0)
    sc = (Xc * N).sum((1, 2)) / (N ** 2).sum()
    lo, hi = np.median(sc[claimed[R]]), np.median(sc[sc > thr])
    thr = 0.5 * (lo + hi)
    print(f"it {it}: flagged {fl.sum()} ({fl.mean():.4f}) | claimed flagged {fl[claimed[R]].sum()} | scores lo {lo:.3f} hi {hi:.3f}")
# residual after removing N: is it exact-ish? centered b - N vs nothing
fl = sc > thr
res = Xc[fl] - N
print("flagged per subject:", {int(s): (int(fl[s25[R] == s].sum()), round(float(fl[s25[R] == s].mean()), 4)) for s in np.unique(s25)})
print("N centered: per-axis std", N.std(0).round(4), "mean", N.mean(0).round(4))
np.savez(os.path.join(D, "p9_rnoise.npz"), N=N, rows=R, flag=fl, score=sc)
