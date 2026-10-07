"""probe 10: right_arm additive patterns: two clusters (score ~1.5 and ~2.5) -> their mean patterns; relation between them;
EM-like multi-pattern assignment"""
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
R = np.flatnonzero(l25 == L); X = A25[R]; Xc = X - X.mean(1, keepdims=True); cl = claimed[R]
P = Xc[~cl].mean(0) - Xc[cl].mean(0)
sc = (Xc * P).sum((1, 2)) / (P ** 2).sum()
A = Xc[(sc > 1.2) & (sc < 1.8)].mean(0); B = Xc[(sc > 2.2) & (sc < 2.8)].mean(0)
print("n A", np.sum((sc > 1.2) & (sc < 1.8)), "n B", np.sum((sc > 2.2) & (sc < 2.8)))
print("norm A", np.linalg.norm(A).round(4), "norm B", np.linalg.norm(B).round(4), "corr(A,B)", (np.sum(A * B) / np.linalg.norm(A) / np.linalg.norm(B)).round(4))
print("A[:6]\n", A[:6]); print("B[:6]\n", B[:6])
# is B a time-warp / sign variant of A? per-axis corr
for k in range(3):
    print("axis", k, "corr A_k,B_k", np.corrcoef(A[:, k], B[:, k])[0, 1].round(3), "std A", A[:, k].std().round(4), "std B", B[:, k].std().round(4))
# per-row nearest pattern among {0, A, B} with free scale? residual energy after subtracting
pats = [np.zeros_like(A), A, B]
E = np.stack([((Xc - p) ** 2).sum((1, 2)) for p in pats], 1)
lab = E.argmin(1)
print("labels (0 none, 1 A, 2 B):", np.bincount(lab, minlength=3), "| claimed labels", np.bincount(lab[cl], minlength=3))
for s in np.unique(s25):
    k = s25[R] == s; print(f"  sbj {s}: n {k.sum()} A {np.sum(lab[k] == 1)} ({np.mean(lab[k] == 1):.4f}) B {np.sum(lab[k] == 2)} ({np.mean(lab[k] == 2):.4f})")
# refine patterns with labels
for it in range(3):
    A = Xc[lab == 1].mean(0) - Xc[lab == 0].mean(0); B = Xc[lab == 2].mean(0) - Xc[lab == 0].mean(0)
    E = np.stack([((Xc - p) ** 2).sum((1, 2)) for p in (np.zeros_like(A), A, B)], 1); lab = E.argmin(1)
    print(f"it {it}: labels {np.bincount(lab, minlength=3)} claimed {np.bincount(lab[cl], minlength=3)}")
d = np.sort(E, 1); gap = (d[:, 1] - d[:, 0]) / (np.linalg.norm(A) ** 2)
print("decision gap quantiles (in |A|^2 units)", np.quantile(gap, [0.01, 0.05, 0.1, 0.5]).round(3))
np.savez(os.path.join(D, "p10_rpat.npz"), A=A, B=B, rows=R, lab=lab, E=E)
