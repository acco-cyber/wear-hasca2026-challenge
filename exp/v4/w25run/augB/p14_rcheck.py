"""probe 14: right_arm types: refit on exact pairs only, show exact M/N, list non-exact residual cases"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=6)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
g = np.load(os.path.join(D, "p13_rglobal.npz")); bad, bk, bj, best, sec = g["bad"], g["bk"], g["bj"], g["best"], g["sec"]
rms = np.sqrt(np.maximum(best, 0) / 150); rms2 = np.sqrt(np.maximum(sec, 0) / 150)
dyn = np.linalg.norm(A26 - A26.mean(1, keepdims=True), axis=(1, 2)) / np.sqrt(150)
for t in (0, 1):
    sel = np.flatnonzero((bk == t) & (rms < 1e-3))
    ai = A26[bad[sel]]; bi = A25[bj[sel]]
    am = ai.mean(0); bm = bi.mean(0)
    M = np.linalg.lstsq((ai - am).reshape(-1, 3), (bi - bm).reshape(-1, 3), rcond=None)[0]; N = bm - am @ M
    r = bi - ai @ M - N
    print(f"type {t}: exact pairs {len(sel)} / {np.sum(bk == t)} | refit max resid {np.abs(r).max():.2e}\nM=\n{M}\nM^T M=\n{M.T @ M}\ndet {np.linalg.det(M):.6f}")
    print(" N[:4]\n", N[:4], "\n N mean", N.mean(0), "std", N.std(0))
    nz = np.flatnonzero(bk == t)
    print(f" rms quantiles all pairs of type: {np.quantile(rms[nz], [0.5, 0.6, 0.7, 0.8, 0.9]).round(4)}")
    np.savez(os.path.join(D, f"p14_type{t}.npz"), M=M, N=N)
# non-exact cases: dyn and rms
ne = rms > 1e-3
print("non-exact matched:", ne.sum(), "of", len(bad), "| dyn of exact median", np.median(dyn[bad[~ne]]).round(4), "non-exact median", np.median(dyn[bad[ne]]).round(4))
print("non-exact rms q", np.quantile(rms[ne], [0.1, 0.5, 0.9]).round(4), "second-best/best ratio q", np.quantile(rms2[ne] / rms[ne], [0.1, 0.5, 0.9]).round(3))
for q in np.flatnonzero(ne)[:5]:
    i = bad[q]; print(f"tile {i} type {bk[q]} rms {rms[q]:.4f} second {rms2[q]:.4f} dyn {dyn[i]:.4f}")
