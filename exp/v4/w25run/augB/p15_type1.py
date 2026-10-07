"""probe 15: right_arm type 1 (signed-perm-like M): residual structure; time warp? scaling? per-pair fit with warp"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=4)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64)
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
g = np.load(os.path.join(D, "p13_rglobal.npz")); bad, bk, bj, best, sec = g["bad"], g["bk"], g["bj"], g["best"], g["sec"]
Ms, Ns = g["Ms"], g["Ns"]
rms = np.sqrt(np.maximum(best, 0) / 150); rms2 = np.sqrt(np.maximum(sec, 0) / 150)
sel = np.flatnonzero(bk == 1)
print("type1 pairs", len(sel), "rms q", np.quantile(rms[sel], [0.1, 0.25, 0.5, 0.75, 0.9]).round(4), "second q", np.quantile(rms2[sel], [0.1, 0.5, 0.9]).round(4))
M1, N1 = Ms[1], Ns[1]
# residual pattern by position averaged over pairs
R = A25[bj[sel]] - A26[bad[sel]] @ M1 - N1
print("mean |resid| by position", np.abs(R).mean((0, 2)).round(4))
# take the 20 lowest-rms dynamic pairs and inspect one
dyn = np.linalg.norm(A26 - A26.mean(1, keepdims=True), axis=(1, 2))
o = sel[np.argsort(rms[sel])]
q = o[len(o) // 4]; i, r = bad[q], bj[q]
a = A26[i]; b = A25[r]
P = -np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])
print("tile", i, "row", r, "rms", rms[q].round(4))
print("a@P[:8]\n", (a @ P)[:8]); print("b[:8]\n", b[:8]); print("b - a@P [:8]\n", (b - a @ P)[:8]); print("N1[:8]\n", N1[:8])
print("b - a@P - N1 [:8]\n", (b - a @ P - N1)[:8])
# ratio test: is b - N0' = s * a@P ? per axis scale
for k in range(3):
    x = (a @ P)[:, k]; y = b[:, k]
    A_ = np.c_[x, np.ones(50)]; cf = np.linalg.lstsq(A_, y, rcond=None)[0]
    print("axis", k, "fit y = s x + c:", cf.round(4), "resid std", (y - A_ @ cf).std().round(4))
np.savez(os.path.join(D, "p15_ex.npz"), a=a, b=b)
