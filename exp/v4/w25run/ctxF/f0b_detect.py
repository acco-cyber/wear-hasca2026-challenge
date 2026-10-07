"""step 0b: detect the augmentation type of every 2025 arm row (range-of-warp test for L / R1, additive-pattern test for
R0/R1), calibrated on the rows that are twins of our tiles (known types); de-augment; replace every twin by our clean tile.
-> cache/w25_clean.npz (acc (48936,50,3) float32, typ per row, anchored rows, row of each test tile)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
np.set_printoptions(linewidth=200, suppress=True, precision=4)
g = np.load(os.path.join(W, "exp", "v4", "w25run", "augB", "aug_params.npz"))
tauL, tauR, M0, N0, P1, N1 = (g[k].astype(np.float64) for k in ("tauL", "tauR", "M0", "N0", "P1", "N1"))


def warp_matrix(tau):
    Wm = np.zeros((50, 50))
    for p, t in enumerate(tau):
        k = min(int(np.floor(t)), 48); f = t - k; Wm[p, k] += 1 - f; Wm[p, k + 1] += f
    return Wm


def inv_interp(tau, Bw):
    """invert b[p] = a(tau[p]) by linear interpolation of the points (tau[p], b[p]) at the integer samples"""
    tu, idx = np.unique(tau, return_index=True); out = np.empty_like(Bw)
    for k in range(Bw.shape[2]):
        out[:, :, k] = np.stack([np.interp(np.arange(50), tu, b[idx]) for b in Bw[:, :, k]])
    return out


WL, WR = warp_matrix(tauL), warp_matrix(tauR)
print("rank WL", np.linalg.matrix_rank(WL), "rank WR", np.linalg.matrix_rank(WR))
PL = np.eye(50) - WL @ np.linalg.pinv(WL); PR = np.eye(50) - WR @ np.linalg.pinv(WR)     # projectors onto the complement of the range
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); sens = st["sensor_test"].astype(int); tsb = st["test_sbj"].astype(int)
tz = np.load(os.path.join(CACHE, "test_twins.npz")); twin, ttyp = tz["twin"], tz["typ"]; assert tz["anchored"].all()
n25 = len(A25); known = np.full(n25, -1); known[twin] = np.where(ttyp == 0, 0, np.where(sens == 3, 1, ttyp))   # 0 clean, 1 L, 1 R0, 2 R1


def rng_resid(P, B):
    return np.sqrt(np.einsum("pq,nqk->npk", P, B) ** 2).mean((1, 2)) / (np.abs(B - B.mean(1, keepdims=True)).mean((1, 2)) + 1e-9)


def rough(B):
    return (np.diff(B, 2, axis=1) ** 2).mean((1, 2))


typ = np.zeros(n25, int)          # 0 clean, 1 L (left arm) / R0 (right arm), 2 R1
# ---- left arm (2025 limb 0): L rows lie in the range of WL
m0 = l25 == 0; rL = rng_resid(PL, A25[m0]); kn = known[m0]
print("left arm range residual: clean", np.quantile(rL[kn == 0], [0, 0.001, 0.01, 0.5]), "| L", np.quantile(rL[kn == 1], [0.5, 0.99, 1]))
thrL = 1e-6; isL = rL < thrL
print(f"  detection on known rows: clean->L {np.mean(isL[kn == 0]):.4f}, L->L {np.mean(isL[kn == 1]):.4f}; all left-arm rows flagged {isL.mean():.4f}")
typ[np.flatnonzero(m0)[isL]] = 1
# ---- right arm (2025 limb 2)
m2 = l25 == 2; B = A25[m2]; kn = known[m2]
r1 = rng_resid(PR, B - N1)
print("right arm R1 range residual: clean", np.quantile(r1[kn == 0], [0, 0.001, 0.01, 0.5]), "| R0", np.quantile(r1[kn == 1], [0, 0.01, 0.5]), "| R1", np.quantile(r1[kn == 2], [0.5, 0.99, 1]))
rb, r0 = rough(B), rough(B - N0); rq = np.log(r0 / rb)
print("right arm log rough(b-N0)/rough(b): clean", np.quantile(rq[kn == 0], [0.5, 0.99, 0.999, 1]), "| R0", np.quantile(rq[kn == 1], [0, 0.001, 0.01, 0.5]), "| R1", np.quantile(rq[kn == 2], [0, 0.5, 1]))
isR1 = r1 < 1e-6
thr0 = 0.5 * (np.quantile(rq[kn == 0], 0.999) + np.quantile(rq[kn == 1], 0.001))
isR0 = (rq < thr0) & ~isR1
print(f"  R0 threshold {thr0:.3f}; known clean -> (R0 {np.mean(isR0[kn == 0]):.4f}, R1 {np.mean(isR1[kn == 0]):.4f}); known R0 -> R0 {np.mean(isR0[kn == 1]):.4f}; known R1 -> R1 {np.mean(isR1[kn == 2]):.4f}")
print(f"  all right-arm rows: R0 {isR0.mean():.4f} R1 {isR1.mean():.4f}")
typ[np.flatnonzero(m2)[isR0]] = 1; typ[np.flatnonzero(m2)[isR1]] = 2
# ---- de-augment (pinv vs interpolation inverse checked on the known pairs)
clean = A25.copy()
iL = np.flatnonzero(m0 & (typ == 1)); iR0 = np.flatnonzero(m2 & (typ == 1)); iR1 = np.flatnonzero(m2 & (typ == 2))
cand = {"pinv": (-np.einsum("pq,nqk->npk", np.linalg.pinv(WL), A25[iL]), np.einsum("pq,nqk->npk", np.linalg.pinv(WR), A25[iR1] - N1) @ P1.T),
        "interp": (-inv_interp(tauL, A25[iL]), inv_interp(tauR, A25[iR1] - N1) @ P1.T)}
o_of = np.full(n25, -1); o_of[twin] = np.arange(len(twin))
for nm, (aL, aR1) in cand.items():
    eL = [np.abs(aL[k] - A26[o_of[r]]).mean() for k, r in enumerate(iL) if o_of[r] >= 0]
    eR = [np.abs(aR1[k] - A26[o_of[r]]).mean() for k, r in enumerate(iR1) if o_of[r] >= 0]
    print(f"inverse {nm}: mean abs err L {np.mean(eL):.4f} R1 {np.mean(eR):.4f}")
best = min(cand, key=lambda nm: np.mean([np.abs(cand[nm][0][k] - A26[o_of[r]]).mean() for k, r in enumerate(iL) if o_of[r] >= 0]))
print("using", best)
clean[iL] = cand[best][0]; clean[iR1] = cand[best][1]; clean[iR0] = (A25[iR0] - N0) @ M0.T
clean[twin] = A26                                                     # our clean tiles replace their twins
print("rows replaced by our tiles", len(twin), "| de-augmented non-twin rows", int(((typ > 0) & (o_of < 0)).sum()))
for L25 in range(4):
    mm = l25 == L25; print(f"2025 limb {L25}: aug share {np.mean(typ[mm] > 0):.4f}; rough clean-version quantiles {np.quantile(rough(clean[mm]), [0.25, 0.5, 0.75]).round(4)}")
np.savez(os.path.join(CACHE, "w25_clean.npz"), acc=clean.astype(np.float32), typ=typ, twin=twin, sbj=s25, limb=l25)
