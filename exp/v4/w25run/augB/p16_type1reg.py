"""probe 16: right_arm type 1: per-position, per-axis regression across pairs b[p,k] = m[p,k] * (a@P)[p,k] + N[p,k]
(magnitude warp test) and time-warp test with the left-arm warp"""
import os, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=4)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64)
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
g = np.load(os.path.join(D, "p13_rglobal.npz")); bad, bk, bj, best = g["bad"], g["bk"], g["bj"], g["best"]
rms = np.sqrt(np.maximum(best, 0) / 150)
sel = np.flatnonzero((bk == 1) & (rms < 0.05))
P = -np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])
X = A26[bad[sel]] @ P; Y = A25[bj[sel]]
print("pairs", len(sel))
m = np.zeros((50, 3)); N = np.zeros((50, 3)); rs = np.zeros((50, 3))
for p in range(50):
    for k in range(3):
        A_ = np.c_[X[:, p, k], np.ones(len(sel))]; cf, *_ = np.linalg.lstsq(A_, Y[:, p, k], rcond=None)
        m[p, k], N[p, k] = cf; rs[p, k] = np.median(np.abs(Y[:, p, k] - A_ @ cf))
print("slope m[p,k] (rows = position)\n", m.T)
print("median abs resid per position\n", rs.max(1))
# also full 3x3 per position: Y[:,p,:] = X[:,p,:] @ Mp + Np
mr = []
for p in range(50):
    A_ = np.c_[X[:, p], np.ones(len(sel))]; cf, *_ = np.linalg.lstsq(A_, Y[:, p], rcond=None)
    mr.append(np.median(np.abs(Y[:, p] - A_ @ cf)))
print("3x3-per-position median abs resid\n", np.array(mr))
np.savez(os.path.join(D, "p16_type1reg.npz"), m=m, N=N)
