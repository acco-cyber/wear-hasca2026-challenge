"""probe 4: inspect endpoint-hit pairs (2026 tile i <-> 2025 row r): sign pattern, time warp structure"""
import os, csv, numpy as np
np.set_printoptions(linewidth=200, suppress=True)
W = r"E:\Claude code\wear"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64)
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
for i, r in ((8, 13810), (44, 23013), (78, 16569), (129, 17768)):
    a = A26[i]; b = A25[r]
    print(f"\n=== 2026 {i} vs 2025 {r}")
    print("a[:8]\n", a[:8].round(4)); print("b[:8]\n", b[:8].round(4))
    print("a[-5:]\n", a[-5:].round(4)); print("b[-5:]\n", b[-5:].round(4))
    # sign / perm fit: best signed permutation via linear lstsq
    M = np.linalg.lstsq(a, b, rcond=None)[0]; print("lstsq M\n", M.round(3), "resid rel", (np.linalg.norm(b - a @ M) / np.linalg.norm(b)).round(4))
    # time warp: for each b sample find tau in a (fractional) under b = -a(tau): search fine grid
    s = -1.0
    tt = np.linspace(0, 49, 4901); ai = np.stack([np.interp(tt, np.arange(50), a[:, k]) for k in range(3)], 1) * s
    tau = []; err = []
    for p in range(50):
        d = np.linalg.norm(ai - b[p], axis=1); j = d.argmin(); tau.append(tt[j]); err.append(d[j])
    print("tau", np.round(tau, 2)); print("err", np.round(err, 3))
