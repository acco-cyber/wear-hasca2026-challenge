"""step 0: verify the 2025 augmentation parameters (augB/aug_params.npz, read-only) on our tiles: every non-exact arm tile
must have an exact augmented twin in the 2025 rows of its subject/limb; check the inverse transforms."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
np.set_printoptions(linewidth=200, suppress=True, precision=4)
g = np.load(os.path.join(W, "exp", "v4", "w25run", "augB", "aug_params.npz"))
for k in g.files:
    print(k, g[k].shape)
tauL, tauR, M0, N0, P1, N1 = (g[k].astype(np.float64) for k in ("tauL", "tauR", "M0", "N0", "P1", "N1"))


def warp_matrix(tau):
    Wm = np.zeros((50, 50))
    for p, t in enumerate(tau):
        k = min(int(np.floor(t)), 48); f = t - k; Wm[p, k] += 1 - f; Wm[p, k + 1] += f
    return Wm


WL, WR = warp_matrix(tauL), warp_matrix(tauR)
print("tauL", tauL[:5], tauL[-5:], "cond WL", np.linalg.cond(WL)); print("tauR", tauR[:5], tauR[-5:], "cond WR", np.linalg.cond(WR))
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); tw = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); sens = st["sensor_test"].astype(int); tsb = st["test_sbj"].astype(int)
d = np.abs(A25[tw] - A26).max((1, 2)); ex = d < 1e-4
aug = {"L": lambda a: np.einsum("pq,nqk->npk", WL, -a), "R0": lambda a: a @ M0 + N0, "R1": lambda a: np.einsum("pq,nqk->npk", WR, a @ P1) + N1}
twin2 = np.where(ex, tw, -1); typ = np.where(ex, 0, -1); res = np.full(len(A26), np.inf)
for Lp, types in ((3, ["L"]), (0, ["R0", "R1"])):
    for s in (22, 23, 24, 25):
        ours = np.flatnonzero((sens == Lp) & (tsb == s) & ~ex); cand = np.flatnonzero((s25 == s) & (l25 == PIPE_TO_25[Lp]))
        B = A25[cand].reshape(len(cand), -1)
        for ti, t in enumerate(types):
            Aa = aug[t](A26[ours]).reshape(len(ours), -1)
            for b0 in range(0, len(ours), 256):
                D_ = np.abs(Aa[b0:b0 + 256, None, :] - B[None]).max(2); j = D_.argmin(1); r = D_[np.arange(len(j)), j]
                ii = ours[b0:b0 + 256]; better = r < res[ii]
                res[ii[better]] = r[better]; twin2[ii[better]] = cand[j[better]]; typ[ii[better]] = ti + 1
        print(f"sbj {s} limb {Lp}: {len(ours)} non-exact; resid quantiles {np.quantile(res[ours], [0.5, 0.9, 0.99, 1])}")
ne = ~ex & np.isfinite(res)
for thr in (1e-4, 1e-3, 1e-2, 0.05):
    print(f"non-exact arm tiles matched with resid < {thr}: {(res[ne] < thr).mean():.4f}")
ok = ex | (res < 1e-3)
print("anchored share (exact or exact-augmented):", ok.mean(), " per limb", [round(float(ok[sens == L].mean()), 4) for L in range(4)])
u, c = np.unique(twin2[ok], return_counts=True); print("duplicate twins among anchored", (c > 1).sum())
# type mix per limb
for Lp in (0, 3):
    mm = (sens == Lp) & ok; print("limb", Lp, "type counts (0 clean, 1.. aug types)", np.bincount(typ[mm], minlength=3))
# inverse transforms
inv = {"L": lambda b: -np.einsum("pq,nqk->npk", np.linalg.pinv(WL), b), "R0": lambda b: (b - N0) @ M0.T,
       "R1": lambda b: np.einsum("pq,nqk->npk", np.linalg.pinv(WR), b - N1) @ P1.T}
for Lp, types in ((3, ["L"]), (0, ["R0", "R1"])):
    for ti, t in enumerate(types):
        mm = (sens == Lp) & ok & (typ == ti + 1)
        rec = inv[t](A25[twin2[mm]]); e = np.abs(rec - A26[mm])
        print(f"inverse {t}: n={mm.sum()} max abs err quantiles {np.quantile(e.max((1, 2)), [0.5, 0.9, 1])}; by position (mean) head {e.mean((0, 2))[:4]} tail {e.mean((0, 2))[-4:]}")
np.savez(os.path.join(CACHE, "test_twins.npz"), twin=twin2, typ=typ, res=res, exact=ex, anchored=ok)
