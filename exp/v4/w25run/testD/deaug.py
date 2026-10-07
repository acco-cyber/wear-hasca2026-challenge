"""Undo the 2025 hosts' augmentation where possible (test side), measured on our tiles' twins:
 left_arm : augmented row b = -Warp(x), Warp = one FIXED time warp (linear map, both endpoints kept: b[0] = -x[0],
            b[49] = -x[49], b[1] == b[0]). Twins of our non-exact left-arm tiles are found exactly by the negated end samples;
            other augmented rows are detected by b[0] == b[1] and inverted with a least-squares inverse warp.
 right_arm: augmented row b = x @ R + white noise (sd ~0.10 g), R a fixed improper rotation. Twins of non-exact tiles by
            minimum rms(b - x R) with a margin; other augmented rows detected by their noise power and un-rotated (b @ R^T),
            the noise stays.
  python deaug.py -> cache/deaug_inputs.npz (A25 de-augmented float32, twin, anch (data order), aug flags, diagnostics)"""
import os, sys, time, csv, json
import numpy as np
from tdlib import *

T0 = time.time()
def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
meta = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in meta]); sens26 = np.array([SENS.index(r["sensor_location"]) for r in meta])
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]; pl25 = MAP25[l25]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin0 = m["twin"]; corr = m["corr"]; margin = m["margin"]
exact = np.abs(A26 - A25[twin0]).max((1, 2)) < 1e-4
twin = np.where(exact, twin0, -1); how = np.where(exact, 1, 0)          # 1 exact, 2 LA end-sample match, 3 RA rotation match
used = np.zeros(len(A25), bool); used[twin0[exact]] = True
info = {}

# ------------------------------------------------------------------ left arm
LA, RA = SENS.index("left_arm"), SENS.index("right_arm")
for s in np.unique(s26):
    r = np.flatnonzero((s25 == s) & (pl25 == LA) & ~used); q = np.flatnonzero(~exact & (sens26 == LA) & (s26 == s))
    f0, f1 = A25[r][:, 0], A25[r][:, -1]
    for i in q:
        k = np.flatnonzero((np.abs(f0 + A26[i][0][None]).max(1) < 1e-6) & (np.abs(f1 + A26[i][-1][None]).max(1) < 1e-6))
        if len(k) == 1:
            twin[i] = r[k[0]]; how[i] = 2
qa = np.flatnonzero(how == 2)
assert len(np.unique(twin[qa])) == len(qa) and not used[twin[qa]].any()
info["LA_nonexact"] = int(np.sum(~exact & (sens26 == LA))); info["LA_matched"] = len(qa)
log(f"left_arm: non-exact {info['LA_nonexact']}, matched by negated end samples {len(qa)}")
# fixed warp: -b = Wf x ; inverse map x = Mi (-b), fitted on the matched pairs (axes pooled), 2-fold CV error reported
Xp = np.concatenate([A26[i].T for i in qa]); Bp = np.concatenate([-A25[twin[i]].T for i in qa])     # (3*len, 50)
Wf = np.linalg.lstsq(Xp, Bp, rcond=None)[0]                                                       # Bp ~ Xp @ Wf
log(f"   forward warp fit rms {np.sqrt(np.mean((Bp - Xp @ Wf) ** 2)):.5f} (signal rms {np.sqrt(np.mean(Bp ** 2)):.3f})")
def fit_inv(Bt, Xt, lam=1e-3):
    G = Bt.T @ Bt + lam * np.eye(50); return np.linalg.solve(G, Bt.T @ Xt)
half = np.arange(len(Xp)) % 2 == 0
e = []
for tr in (half, ~half):
    Mi = fit_inv(Bp[tr], Xp[tr]); e.append(np.sqrt(np.mean((Bp[~tr] @ Mi - Xp[~tr]) ** 2)))
Mi = fit_inv(Bp, Xp)
info["LA_inverse_cv_rms"] = float(np.mean(e)); log(f"   inverse warp 2-fold CV rms {np.mean(e):.5f}; per-sample CV rms at edges",
    np.round(np.sqrt(np.mean((Bp[~half] @ fit_inv(Bp[half], Xp[half]) - Xp[~half]) ** 2, 0))[[0, 1, 2, 47, 48, 49]], 4).tolist())
eq01 = np.abs(A25[:, 0] - A25[:, 1]).max(1) < 1e-7
info["LA_detector_on_matched_twins"] = float(eq01[twin[qa]].mean()); info["LA_detector_on_exact_twins"] = float(eq01[twin[exact & (sens26 == LA)]].mean())
info["LA_detector_on_ours"] = float((np.abs(A26[:, 0] - A26[:, 1]).max(1) < 1e-7).mean())
augLA = (pl25 == LA) & eq01
log(f"   detector b0==b1: matched augmented twins {info['LA_detector_on_matched_twins']:.4f}, exact twins {info['LA_detector_on_exact_twins']:.4f}, "
    f"our clean tiles {info['LA_detector_on_ours']:.4f}; all left_arm rows flagged {augLA[pl25 == LA].mean():.4f}")

# ------------------------------------------------------------------ right arm
k = np.flatnonzero(~exact & (corr > 0.99) & (margin > 0.05) & (sens26 == RA))
res = np.array([np.linalg.lstsq(A26[i], A25[twin0[i]], rcond=None)[1].sum() for i in k])
kk = k[res < np.quantile(res, 0.8)]                                                                # robust: drop the worst 20%
U, _, Vt = np.linalg.svd(np.concatenate([A26[i] for i in kk]).T @ np.concatenate([A25[twin0[i]] for i in kk])); R = U @ Vt
sd = float(np.median([(A25[twin0[i]] - A26[i] @ R).std() for i in kk]))
log(f"right_arm: R (det {np.linalg.det(R):.3f}) from {len(kk)} confident pairs, noise sd {sd:.4f}\n{np.round(R, 4)}")
def d2(X):
    return (np.diff(X, 2, axis=1) ** 2).sum(2).mean(1)
for s in np.unique(s26):
    r = np.flatnonzero((s25 == s) & (pl25 == RA) & ~used); q = np.flatnonzero(~exact & (sens26 == RA) & (s26 == s))
    XR = np.einsum("ntc,cd->ntd", A26[q], R)
    D = np.stack([np.sqrt(((XR[j][None] - A25[r]) ** 2).mean((1, 2))) for j in range(len(q))])
    b = D.argmin(1); best = D.min(1); D[np.arange(len(q)), b] = 9; sec = D.min(1)
    ok = (best < 1.6 * sd) & (sec > best + 0.5 * sd)
    for j in np.flatnonzero(ok):
        twin[q[j]] = r[b[j]]; how[q[j]] = 3
qr = np.flatnonzero(how == 3)
u, c = np.unique(twin[qr], return_counts=True); dup = np.isin(twin[qr], u[c > 1])
how[qr[dup]] = 0; twin[qr[dup]] = -1; qr = qr[~dup]
log(f"   first pass: matched by rotation (unique, margin) {len(qr)} of {int(np.sum(~exact & (sens26 == RA)))}")
# the 'noise' is ONE fixed (50,3) pattern N0 (pairwise residual corr ~0.995): b = x @ A + N0. Joint LS fit on the matches.
Xq = np.stack([A26[i] for i in qr]); Bq = np.stack([A25[twin[i]] for i in qr]); A_ = R.copy()
for it in range(20):
    N0 = (Bq - Xq @ A_).mean(0)
    A_ = np.linalg.lstsq(Xq.reshape(-1, 3), (Bq - N0[None]).reshape(-1, 3), rcond=None)[0]
fit = np.sqrt(np.mean((Bq - Xq @ A_ - N0[None]) ** 2))
log(f"   fixed-pattern model b = x A + N0: fit rms {fit:.6f}; N0 rms {np.sqrt(np.mean(N0 ** 2)):.4f}; A det {np.linalg.det(A_):.4f}\n{np.round(A_, 4)}")
Ainv = np.linalg.inv(A_)
def undo_ra(B):
    return (B - N0[None]) @ Ainv
# re-match every non-exact right-arm tile exactly: undo_ra(b) == x
def rematch():
    how[(how == 3)] = 0; twin[(how == 0) & ~exact & (sens26 == RA)] = -1
    for s in np.unique(s26):
        r = np.flatnonzero((s25 == s) & (pl25 == RA) & ~used); q = np.flatnonzero(~exact & (sens26 == RA) & (s26 == s))
        Ur = undo_ra(A25[r])
        for i in q:
            D = np.sqrt(((Ur - A26[i][None]) ** 2).mean((1, 2))); j = D.argmin(); D[j] = 9
            if np.sqrt(((Ur[j] - A26[i]) ** 2).mean()) < 0.005 and D.min() > 0.003:
                twin[i] = r[j]; how[i] = 3
    qr = np.flatnonzero(how == 3); u, c = np.unique(twin[qr], return_counts=True); dup = np.isin(twin[qr], u[c > 1])
    how[qr[dup]] = 0; twin[qr[dup]] = -1
    return qr[~dup]
qr = rematch()
Xq = np.stack([A26[i] for i in qr]); Bq = np.stack([A25[twin[i]] for i in qr])                 # refit on the tight matches
for it in range(20):
    N0 = (Bq - Xq @ A_).mean(0)
    A_ = np.linalg.lstsq(Xq.reshape(-1, 3), (Bq - N0[None]).reshape(-1, 3), rcond=None)[0]
Ainv = np.linalg.inv(A_); fit = np.sqrt(np.mean((Bq - Xq @ A_ - N0[None]) ** 2))
log(f"   refit on {len(qr)} tight matches: rms {fit:.6f}")
qr = rematch()
info["RA_nonexact"] = int(np.sum(~exact & (sens26 == RA))); info["RA_matched"] = len(qr); info["RA_fixed_pattern_fit_rms"] = float(fit)
log(f"   right_arm non-exact {info['RA_nonexact']}, matched exactly after undoing (max|diff| < 1e-3, unique) {len(qr)}")
dd = d2(A25); du = d2(undo_ra(A25)); nz = d2(N0[None])[0]
augRA = (pl25 == RA) & (dd - du > 0.5 * nz)                                                     # removing N0 makes the row smooth
info["RA_flag_rate_exact_twins"] = float(augRA[twin[exact & (sens26 == RA)]].mean()); info["RA_flag_rate_matched"] = float(augRA[twin[qr]].mean())
log(f"   detector d2(b) - d2(undo(b)) > 0.5 d2(N0): matched augmented twins {info['RA_flag_rate_matched']:.4f}, exact (clean) twins "
    f"{info['RA_flag_rate_exact_twins']:.4f}; all right_arm rows flagged {augRA[pl25 == RA].mean():.4f}")

# ------------------------------------------------------------------ de-augmented 2025 array
A = A25.copy()
iL = np.flatnonzero(augLA); A[iL] = np.einsum("tk,nkc->ntc", Mi.T, -A25[iL])          # x = (-b) @ Mi per axis
iR = np.flatnonzero(augRA); A[iR] = undo_ra(A25[iR])
# sanity on matched twins: de-augmented row vs our clean tile
for nm, qq in (("LA", qa), ("RA", qr)):
    if len(qq):
        er = np.sqrt(((A[twin[qq]] - A26[qq]) ** 2).mean((1, 2))); e0 = np.abs(A[twin[qq]][:, [0, -1]] - A26[qq][:, [0, -1]]).max((1, 2))
        log(f"   {nm}: de-augmented twin vs our tile rms median {np.median(er):.4f}, end-sample max|diff| median {np.median(e0):.4f}")
anch = how > 0
log(f"anchored {anch.mean():.4f} (exact {exact.mean():.4f}); per limb", {SENS[L]: round(float(anch[sens26 == L].mean()), 4) for L in range(4)})
assert (pl25[twin[anch]] == sens26[anch]).all() and (s25[twin[anch]] == s26[anch]).all() and len(np.unique(twin[anch])) == anch.sum()
info.update(anchored=float(anch.mean()), anchored_per_limb={SENS[L]: float(anch[sens26 == L].mean()) for L in range(4)},
            aug_share_rows={"left_arm": float(augLA[pl25 == LA].mean()), "right_arm": float(augRA[pl25 == RA].mean())}, R=R.tolist(), noise_sd=sd)
np.savez(os.path.join(CACHE, "deaug_inputs.npz"), A25=A.astype(np.float32), twin=twin, anch=anch, how=how, augLA=augLA, augRA=augRA, R=R, Mi=Mi, sd=sd,
         A_ra=A_, N0=N0, Wf=Wf)
json.dump(info, open(os.path.join(HERE, "deaug_info.json"), "w"), indent=1)
log("saved")

