"""The 2025 WEAR test-set augmentation, reverse-engineered from twin pairs (our clean 2026 tile a <-> 2025 row b).

All parameters are FIXED (the same warp / matrix / noise pattern for every augmented row; the hosts evidently drew them
once).  Exact to float32 precision on every twin pair:
    kind 0 "L"  (left_arm only, 25 % of rows):        b = -warp(a, tauL)
    kind 1 "R0" (right_arm only, ~20 %):               b = a @ M0 + N0        M0 orthogonal, det -1 (a reflection)
    kind 2 "R1" (right_arm only, ~20 %):               b = warp(a, tauR) @ P1 + N1   P1 = -(swap x,y)
legs: never augmented.  warp(a, tau)[p] = linear interpolation of a at the fractional sample index tau[p]
(tau[0] = 0, tau[49] = 49: end samples are kept, the interior is shifted by up to ~3.3 samples).
N0, N1: fixed (50, 3) additive patterns, ~0.1 g per axis.

API (tiles are (n, 50, 3) float arrays in g, sample order as in the data files):
    augment(tiles, rng, rate=None, limb=None, exact_count=True) -> (out, kind)      kind -1 = untouched
    detect(tiles, limb=None) -> (kind, score)   label-free: exact linear constraints of the warps + noise matched filter
    restore(tiles, kind) -> clean estimate (R0 exact; L/R1 exact up to the 3-4 directions the warps destroy, filled
                            by a smoothness prior; the first/last 3 samples are recovered exactly)
limb may be a name ("left_arm", "right_arm", "left_leg", "right_leg") or the pipeline sensor index
(SENS order 0 right_arm, 1 right_leg, 2 left_leg, 3 left_arm) or None (= every kind allowed)."""
import os
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
_P = np.load(os.path.join(_D, "aug_params.npz"))
TAU_L, TAU_R = _P["tauL"], _P["tauR"]
M0, N0, P1, N1 = _P["M0"], _P["N0"], _P["P1"], _P["N1"]
KINDS = ("L", "R0", "R1")
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]
# measured on the 2025 file (detect() over all 48,936 rows; see report): per-limb rate of each kind
RATES = {"left_arm": {0: 0.25}, "right_arm": {1: 0.20, 2: 0.20}, "left_leg": {}, "right_leg": {}}


def _limb_name(limb):
    if limb is None:
        return None
    if isinstance(limb, (int, np.integer)):
        return SENS[int(limb)]
    return str(limb)


def warp_matrix(tau):
    Wm = np.zeros((50, 50))
    for p, t in enumerate(tau):
        k = min(int(np.floor(t)), 48); f = t - k
        Wm[p, k] += 1 - f; Wm[p, k + 1] += f
    return Wm


WL, WR = warp_matrix(TAU_L), warp_matrix(TAU_R)


def _left_null(Wm, tol=1e-9):
    U, S, _ = np.linalg.svd(Wm)
    return U[:, S < tol * S[0]]                 # (50, d): v^T (W a) == 0 for every a


VL, VR = _left_null(WL), _left_null(WR)        # d = 4 (L), 3 (R1)


def _reg_inverse(Wm, tol=1e-9):
    """linear operator c -> a: exact pseudo-inverse on the observed subspace of W (keeps weakly observed directions,
    e.g. the head of warpR), plus the minimum-roughness (second-difference) fill of the exact null space of W"""
    U, S, Vt = np.linalg.svd(Wm)
    keep = S > tol * S[0]
    Pinv = (Vt[keep].T / S[keep]) @ U[:, keep].T            # (50, 50)
    Vn = Vt[~keep].T                                        # (50, d) null space of W
    if Vn.shape[1] == 0:
        return Pinv
    D2 = np.diff(np.eye(50), 2, axis=0); G = D2.T @ D2
    Z = -np.linalg.solve(Vn.T @ G @ Vn, Vn.T @ G)           # null coefficients minimising ||D2 (a0 + Vn z)||
    return (np.eye(50) + Vn @ Z) @ Pinv


IL, IR = _reg_inverse(WL), _reg_inverse(WR)


def apply_kind(a, k):
    """a (n,50,3) -> augmented copy of kind k (0 L, 1 R0, 2 R1)"""
    a = np.asarray(a, np.float64)
    if k == 0:
        return -np.einsum("pq,nqc->npc", WL, a)
    if k == 1:
        return a @ M0 + N0
    if k == 2:
        return np.einsum("pq,nqc->npc", WR, a) @ P1 + N1
    raise ValueError(k)


def augment(tiles, rng, rate=None, limb=None, exact_count=True):
    """Reproduce the 2025 augmentation.  rate: total fraction augmented (default: the measured per-limb rate; for
    limb=None the 2025 overall rate 0.1625 with the three kinds equally likely).  Returns (out float32, kind int8)."""
    X = np.asarray(tiles, np.float64); n = len(X)
    name = _limb_name(limb)
    if name is None:
        shares = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}; tot = 0.1625 if rate is None else float(rate)
    else:
        r = RATES[name]; t0 = sum(r.values())
        if t0 == 0:
            if rate:                                # legs were never augmented in 2025; allow forcing any kind
                shares = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}; tot = float(rate)
            else:
                return X.astype(np.float32), np.full(n, -1, np.int8)
        else:
            shares = {k: v / t0 for k, v in r.items()}; tot = t0 if rate is None else float(rate)
    kind = np.full(n, -1, np.int8)
    if exact_count:
        m = int(round(tot * n)); idx = rng.permutation(n)[:m]
        ks = np.array(list(shares)); pk = np.array([shares[k] for k in ks])
        cnt = np.floor(pk * m).astype(int); cnt[: m - cnt.sum()] += 1
        kind[idx] = np.repeat(ks, cnt)[rng.permutation(m)] if m else kind[idx]
    else:
        u = rng.random(n); hit = u < tot
        ks = np.array(list(shares)); pk = np.array([shares[k] for k in ks])
        kind[hit] = rng.choice(ks, size=hit.sum(), p=pk)
    out = X.copy()
    for k in range(3):
        j = np.flatnonzero(kind == k)
        if len(j):
            out[j] = apply_kind(X[j], k)
    return out.astype(np.float32), kind


def noise_stat(X, N, d=3):
    """matched filter for a fixed additive pattern N on the d-th differences (signal energy is low-frequency, the
    pattern is white): ~1 when N is present, ~0 otherwise; '> 0.5' <=> removing N lowers the HF energy"""
    HN = np.diff(N, d, axis=0); HB = np.diff(np.asarray(X, np.float64), d, axis=1)
    return (HB * HN).sum((1, 2)) / (HN ** 2).sum()


def detect(tiles, limb=None, tol=2e-5, thr=0.5):
    """Label-free detection.  Returns kind (-1 clean, 0 L, 1 R0, 2 R1) and a (n, 4) score matrix
    [resid_L, resid_R1, stat_N0, stat_N1].  Rules (in this order):
      L : the 4 exact linear constraints VL^T b == 0 (left null space of warpL; relative residual < tol, float32 ~1e-7)
      R1: the 3x3 exact constraints VR^T ((b - N1) @ P1^T) == 0 and the N1 matched filter > thr
      R0: no exact constraint exists (orthogonal map + noise): N0 matched filter (3rd differences) > thr.
    With limb given only that limb's kinds are allowed (recommended: with limb=None the N0 filter fires on ~0.5 % of
    very dynamic leg tiles).  On the 2025 twins: 12,234/12,234 correct with limb given (see validate.py)."""
    X = np.asarray(tiles, np.float64); n = len(X)
    name = _limb_name(limb)
    scale = np.sqrt((X ** 2).sum((1, 2)) / 150) + 1e-9
    rL = np.sqrt((np.einsum("pd,npc->ndc", VL, X) ** 2).sum((1, 2))) / scale
    C1 = (X - N1) @ P1.T
    r1 = np.sqrt((np.einsum("pd,npc->ndc", VR, C1) ** 2).sum((1, 2))) / scale
    t0 = noise_stat(X, N0); t1 = noise_stat(X, N1)
    kind = np.full(n, -1, np.int8)
    allow = {0, 1, 2} if name is None else set(RATES[name])
    if 0 in allow:
        kind[rL < tol] = 0
    if 2 in allow:
        kind[(kind < 0) & (r1 < tol) & (t1 > thr)] = 2
    if 1 in allow:
        kind[(kind < 0) & (t0 > thr)] = 1
    return kind, np.stack([rL, r1, t0, t1], 1)


def restore(tiles, kind):
    """invert the augmentation of each row given its kind (-1 rows are returned unchanged)"""
    X = np.asarray(tiles, np.float64).copy(); kind = np.asarray(kind)
    j = np.flatnonzero(kind == 0)
    if len(j):
        X[j] = -np.einsum("pq,nqc->npc", IL, X[j])
    j = np.flatnonzero(kind == 1)
    if len(j):
        X[j] = (X[j] - N0) @ M0.T
    j = np.flatnonzero(kind == 2)
    if len(j):
        X[j] = np.einsum("pq,nqc->npc", IR, (X[j] - N1) @ P1.T)
    return X.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(0, 0.02, (2000, 50, 3)), 1) + rng.normal(0, 1, (2000, 1, 3))
    for lb in ("left_arm", "right_arm", "left_leg"):
        b, k = augment(a, rng, limb=lb)
        kd, _ = detect(b, limb=lb)
        print(lb, "augmented", np.bincount(k + 1, minlength=4), "detected", np.bincount(kd + 1, minlength=4), "acc", np.mean(kd == k).round(4),
              "| restore max err (aug rows)", np.abs(restore(b, kd) - a)[k >= 0].max() if (k >= 0).any() else 0)
