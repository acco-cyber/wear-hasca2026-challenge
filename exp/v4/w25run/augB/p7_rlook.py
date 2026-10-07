"""probe 7: right_arm non-exact: centered magnitude correlation (with/without the left-arm warp) against unclaimed 2025
right_arm rows; print top matches and one example pair"""
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
L = 2
bad = np.flatnonzero(~ex & (l26 == L))
def zs(x):
    x = x - x.mean(-1, keepdims=True); return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
M25 = np.linalg.norm(A25, axis=2); M26 = np.linalg.norm(A26, axis=2); MW26 = M26 @ Wop.T
dyn = M26.std(1)
for nm, Q in (("plain", M26), ("warp", MW26)):
    tops = []
    for s in np.unique(s26[bad]):
        q = bad[s26[bad] == s]; c = np.flatnonzero((s25 == s) & (l25 == L) & ~claimed)
        C = zs(Q[q]) @ zs(M25[c]).T; o = np.sort(C, 1)[:, ::-1][:, :2]; tops.append(o)
    tops = np.concatenate(tops)
    print(f"[{nm}] best corr q {np.quantile(tops[:, 0], [0.1, 0.25, 0.5, 0.75, 0.9]).round(4)} | 2nd {np.quantile(tops[:, 1], [0.1, 0.5, 0.9]).round(4)}")
# dynamic example
i = bad[np.argsort(-dyn[bad])[len(bad) // 3]]
s = s26[i]; c = np.flatnonzero((s25 == s) & (l25 == L) & ~claimed)
C = zs(MW26[i]) @ zs(M25[c]).T; o = np.argsort(-C)[:3]
print("example 2026 row", i, "dyn", dyn[i].round(3), "top warp-mag corr", C[o].round(4), "rows", c[o])
a = A26[i]; b = A25[c[o[0]]]
print("a mag", M26[i].round(3)); print("b mag", M25[c[o[0]]].round(3))
print("a[:6]\n", a[:6].round(4)); print("b[:6]\n", b[:6].round(4)); print("a[-4:]\n", a[-4:].round(4)); print("b[-4:]\n", b[-4:].round(4))
print("mean a", a.mean(0).round(3), "mean b", b.mean(0).round(3))
Mfit = np.linalg.lstsq(np.c_[Wop @ a, np.ones(50)], b, rcond=None)[0]; print("fit [warp a,1] -> b\n", Mfit.round(3))
r = b - np.c_[Wop @ a, np.ones(50)] @ Mfit; print("resid rel (centered)", (np.linalg.norm(r) / np.linalg.norm(b - b.mean(0))).round(4))
U, S, Vt = np.linalg.svd(Mfit[:3]); print("singular values", S.round(4), "det", np.linalg.det(Mfit[:3]).round(4))
