"""probe 12: right_arm pairs under b - N_k ~ [a, 1] @ M (k in none/A/B), residual normalised by centred energy of b - N_k;
then inspect M of confident pairs (rotation? scale?)"""
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
pz = np.load(os.path.join(D, "p10_rpat.npz")); pats = [np.zeros((50, 3)), pz["A"], pz["B"]]
L = 2
bad = np.flatnonzero(~ex & (l26 == L))
dyn = np.linalg.norm(A26 - A26.mean(1, keepdims=True), axis=(1, 2))
r1 = np.full((len(bad), 3), np.nan); r2 = np.full((len(bad), 3), np.nan); arg = np.full((len(bad), 3), -1)
for s in np.unique(s26[bad]):
    qq = np.flatnonzero(s26[bad] == s); c = np.flatnonzero((s25 == s) & (l25 == L) & ~claimed)
    for k in range(3):
        Bk = A25[c] - pats[k]; Bkc = Bk - Bk.mean(1, keepdims=True); En = (Bkc ** 2).sum((1, 2))
        for q in qq:
            a = A26[bad[q]]; ac = a - a.mean(0)
            Q, _ = np.linalg.qr(ac)                                   # offset handled by centring
            pr = np.einsum("pk,npj->nkj", Q, Bkc); rr = (En - (pr ** 2).sum((1, 2))) / En
            o = np.argpartition(rr, 2)[:2]; o = o[np.argsort(rr[o])]
            r1[q, k], r2[q, k], arg[q, k] = rr[o[0]], rr[o[1]], c[o[0]]
kb = r1.argmin(1); rb = r1[np.arange(len(bad)), kb]; r2b = r2[np.arange(len(bad)), kb]
print("best pattern counts:", np.bincount(kb, minlength=3))
print("best resid q", np.quantile(rb, [0.1, 0.25, 0.5, 0.75, 0.9]).round(4), "| second q", np.quantile(r2b, [0.1, 0.5, 0.9]).round(4))
dq = dyn[bad] > np.median(dyn[bad])
print("dynamic half: best", np.quantile(rb[dq], [0.1, 0.5, 0.9]).round(4), "second", np.quantile(r2b[dq], [0.1, 0.5, 0.9]).round(4))
conf = np.flatnonzero((rb < 0.2 * r2b) & dq)
print("confident pairs", len(conf))
for q in conf[:8]:
    k = kb[q]; a = A26[bad[q]]; b = A25[arg[q, k]] - pats[k]
    M = np.linalg.lstsq(np.c_[a, np.ones(50)], b, rcond=None)[0]
    U, S, Vt = np.linalg.svd(M[:3])
    print(f"tile {bad[q]} pattern {k} resid {rb[q]:.4f} vs {r2b[q]:.4f} | M\n{M}\n  sv {S.round(4)} det {np.linalg.det(M[:3]):.4f}")
np.savez(os.path.join(D, "p12_rlin.npz"), bad=bad, r1=r1, r2=r2, arg=arg)
