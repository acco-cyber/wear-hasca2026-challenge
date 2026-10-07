"""Fit the exact fixed parameters of the three 2025 augmentation types from twin pairs (2026 clean tile a <-> 2025 row b)
   L  (left_arm):  b = -warpL(a)
   R0 (right_arm): b = a @ M0 + N0               (M0 orthogonal, det -1)
   R1 (right_arm): b = warpR(a) @ P1 + N1         (P1 signed permutation)
warp(a)[p] = linear interpolation of a at fractional sample tau[p].  Writes aug_params.npz next to this file."""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=6)
W = r"E:\Claude code\wear"; D = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64)
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
p5 = np.load(os.path.join(D, "p5_pairs.npz")); pL = p5["pairs"]
g = np.load(os.path.join(D, "p13_rglobal.npz")); bad, bk, bj, best = g["bad"], g["bk"], g["bj"], g["best"]
rms = np.sqrt(np.maximum(best, 0) / 150)


def solve_warp(X, Y, tau0, with_noise):
    """per position p: given the bracket k = floor(tau0[p]), solve f (and N[p]) exactly by least squares across pairs"""
    n = len(X); tau = np.zeros(50); N = np.zeros((50, 3))
    for p in range(50):
        best = None
        for k in sorted({min(int(np.floor(tau0[p])), 48), max(min(int(np.floor(tau0[p])) - 1, 48), 0), min(int(np.ceil(tau0[p])), 48)}):
            d = (X[:, k + 1] - X[:, k]).ravel(); y = (Y[:, p] - X[:, k]).ravel()
            cols = [d] + ([np.tile(np.eye(3), (n, 1))[:, j] for j in range(3)] if with_noise else [])
            A_ = np.stack(cols, 1); cf = np.linalg.lstsq(A_, y, rcond=None)[0]
            f = float(np.clip(cf[0], 0, 1)); nn = cf[1:4] if with_noise else np.zeros(3)
            r = np.abs(y - f * d - (np.tile(nn, n) if with_noise else 0)).max()
            if best is None or r < best[0]:
                best = (r, k + f, nn)
        tau[p] = best[1]; N[p] = best[2]
    return tau, N


def warp_matrix(tau):
    Wm = np.zeros((50, 50))
    for p, t in enumerate(tau):
        k = min(int(np.floor(t)), 48); f = t - k; Wm[p, k] += 1 - f; Wm[p, k + 1] += f
    return Wm


# --- L
XL = -A26[pL[:, 0]]; YL = A25[pL[:, 1]]
tauL, _ = solve_warp(XL, YL, np.median(p5["tau"], 0), False)
WL = warp_matrix(tauL); rL = np.abs(YL - np.einsum("pq,nqk->npk", WL, XL)).max((1, 2))
print("L: pairs", len(pL), "max abs resid quantiles", np.quantile(rL, [0.5, 0.99, 1]))
# --- R0
s0 = np.flatnonzero((bk == 0) & (rms < 1e-3)); a0 = A26[bad[s0]]; b0 = A25[bj[s0]]
am, bm = a0.mean(0), b0.mean(0)
M0 = np.linalg.lstsq((a0 - am).reshape(-1, 3), (b0 - bm).reshape(-1, 3), rcond=None)[0]
U, _, Vt = np.linalg.svd(M0); M0o = U @ Vt                                   # project to the orthogonal group
N0 = np.median(b0 - a0 @ M0o, 0)
r0 = np.abs(b0 - a0 @ M0o - N0).max((1, 2))
print("R0: pairs", len(s0), "M0 (orthogonalised)\n", M0o, "\n det", np.linalg.det(M0o), "| max abs resid quantiles", np.quantile(r0, [0.5, 0.99, 1]))
# --- R1
P1 = -np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])
t17 = np.load(os.path.join(D, "p17_type1warp.npz"))
s1 = np.flatnonzero((bk == 1) & (rms < 0.05)); X1 = A26[bad[s1]] @ P1; Y1 = A25[bj[s1]]
tauR, N1 = solve_warp(X1, Y1, t17["tau"], True)
WR = warp_matrix(tauR); r1 = np.abs(Y1 - np.einsum("pq,nqk->npk", WR, X1) - N1).max((1, 2))
print("R1: pairs", len(s1), "max abs resid quantiles", np.quantile(r1, [0.5, 0.9, 0.99, 1]))
# R1 with P1 applied after or before warp is the same (linear); N1 is additive after the warp+perm
print("tauL\n", tauL.round(5)); print("tauR\n", tauR.round(5))
print("rank WL", np.linalg.matrix_rank(WL), "rank WR", np.linalg.matrix_rank(WR))
print("N0 mean/std", N0.mean(0).round(4), N0.std(0).round(4), "| N1 mean/std", N1.mean(0).round(4), N1.std(0).round(4))
# relation between N0 and N1 (same seed noise?)
print("corr(N0, N1)", np.corrcoef(N0.ravel(), N1.ravel())[0, 1].round(4), "corr(N0@M0^T, N1@P1^T)", np.corrcoef((N0 @ M0o.T).ravel(), (N1 @ P1.T).ravel())[0, 1].round(4))
np.savez(os.path.join(D, "aug_params.npz"), tauL=tauL, tauR=tauR, M0=M0o, N0=N0, P1=P1, N1=N1)
print("saved", os.path.join(D, "aug_params.npz"))
