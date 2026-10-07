"""probe 17: right_arm type 1: joint per-position search of the warp tau_p and the fixed additive N[p] across pairs:
b_i[p] = (a_i @ P)(tau_p) + N[p]"""
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
tau = np.zeros(50); N = np.zeros((50, 3)); cost = np.zeros(50)
for p in range(50):
    grid = np.arange(max(0, p - 4), min(49, p + 4) + 1e-9, 0.0005)
    best_c = np.inf
    for t in grid:
        k = min(int(np.floor(t)), 48); f = t - k
        xi = X[:, k] * (1 - f) + X[:, k + 1] * f
        r = Y[:, p] - xi; n = np.median(r, 0); c = np.median(np.abs(r - n).max(1))
        if c < best_c:
            best_c = c; tau[p] = t; N[p] = n
    cost[p] = best_c
print("tau\n", tau.round(4)); print("tau - p\n", (tau - np.arange(50)).round(4)); print("median max-abs residual per position\n", cost)
Wop = np.zeros((50, 50))
for p, t in enumerate(tau):
    k = min(int(np.floor(t)), 48); f = t - k; Wop[p, k] += 1 - f; Wop[p, k + 1] += f
R = Y - np.einsum("pq,nqk->npk", Wop, X) - N
mx = np.abs(R).max((1, 2))
print("per-pair max abs resid quantiles", np.quantile(mx, [0.25, 0.5, 0.75, 0.9]).round(5))
np.savez(os.path.join(D, "p17_type1warp.npz"), tau=tau, N=N, P=P)
