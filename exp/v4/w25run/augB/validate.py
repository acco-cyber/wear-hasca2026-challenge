"""Validate aug.py on the real data:
 (1) exact re-match of every non-exact 2026 tile to its augmented 2025 row under the fitted kinds
 (2) detect() vs truth on the 12,234 twin rows; false positives on our clean 2026 tiles
 (3) detect() over all 48,936 2025 rows -> augmented fraction per (subject, limb)
 (4) restore() error on true pairs (all samples / boundary samples)
 (5) per-pair transform models (linear 3x3, scale-only, signed perm, Procrustes) with / without the true warp
 (6) replaceable fraction.  Saves validate.npz (twin25: 2026 row -> exact 2025 row incl. augmented, kind25 per 2025 row)."""
import os, csv, itertools, json, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=5)
W = r"E:\Claude code\wear"; D = os.path.dirname(os.path.abspath(__file__))
import aug
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]          # w25.npz limb codes (verified via exact twins)
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float32)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
out = {}
# (1) exact re-match
twin25 = np.where(ex, twin, -1); kind_t = np.where(ex, -1, -9)
for i in np.flatnonzero(~ex):
    name = LIMBS[l26[i]]; c = np.flatnonzero((s25 == s26[i]) & (l25 == l26[i]))
    for k in aug.RATES[name]:
        b = aug.apply_kind(A26[i:i + 1], k)[0]
        d = np.abs(A25[c] - b).max((1, 2)); j = d.argmin()
        if d[j] < 1e-4:
            twin25[i] = c[j]; kind_t[i] = k; break
print("(1) non-exact tiles", (~ex).sum(), "-> exact augmented twin found", np.sum((~ex) & (twin25 >= 0)), "| by kind", {aug.KINDS[k]: int(np.sum(kind_t == k)) for k in range(3)})
print("    twin25 unique", len(np.unique(twin25[twin25 >= 0])), "of", np.sum(twin25 >= 0))
out["matched_nonexact"] = int(np.sum((~ex) & (twin25 >= 0)))
# (2) detection vs truth on twin rows
ok = twin25 >= 0
kd25 = np.full(len(A25), -9, np.int8); sc25 = np.zeros((len(A25), 4))
for L in range(4):
    j = np.flatnonzero(l25 == L); kd25[j], sc25[j] = aug.detect(A25[j], limb=LIMBS[L])
kd_all = np.full(len(A25), -9, np.int8)
for L in range(4):
    j = np.flatnonzero(l25 == L); kd_all[j], _ = aug.detect(A25[j], limb=None)   # limb-agnostic variant
truth = kind_t[ok]; pred = kd25[twin25[ok]]; pred_any = kd_all[twin25[ok]]
print("(2) detect (limb-aware) accuracy on twin rows", np.mean(pred == truth).round(5), "| limb-agnostic", np.mean(pred_any == truth).round(5))
cm = np.zeros((4, 4), int)
for t, p in zip(truth, pred):
    cm[t + 1, p + 1] += 1
print("    confusion rows=truth (clean,L,R0,R1) cols=pred\n", cm)
cma = np.zeros((4, 4), int)
for t, p in zip(truth, pred_any):
    cma[t + 1, p + 1] += 1
print("    limb-agnostic confusion\n", cma)
fp26, _ = aug.detect(A26, limb=None)
print("    false positives on clean 2026 tiles (limb-agnostic):", int(np.sum(fp26 >= 0)), "of", len(A26))
out["detect_acc_twins"] = float(np.mean(pred == truth)); out["detect_acc_twins_limb_agnostic"] = float(np.mean(pred_any == truth))
out["fp_clean2026"] = int(np.sum(fp26 >= 0))
# (3) augmented fraction per (subject, limb)
print("(3) detected kinds per (subject, limb) on all 2025 rows  [n, L, R0, R1, frac]")
frac = {}
for s in np.unique(s25):
    for L in range(4):
        k = (s25 == s) & (l25 == L); c = [int(np.sum(kd25[k] == q)) for q in range(3)]
        frac[f"{s}_{LIMBS[L]}"] = round(sum(c) / k.sum(), 4)
        print(f"    sbj {s} {LIMBS[L]:9s} n {k.sum():5d} L {c[0]:5d} R0 {c[1]:5d} R1 {c[2]:5d} frac {sum(c) / k.sum():.4f}  (R0 {c[1] / k.sum():.4f} R1 {c[2] / k.sum():.4f})")
tot = np.mean(kd25 >= 0); print("    overall augmented fraction", tot.round(5), "| per limb", {LIMBS[L]: round(float(np.mean(kd25[l25 == L] >= 0)), 4) for L in range(4)})
out["aug_fraction"] = float(tot); out["aug_fraction_by_sbj_limb"] = frac
out["kind_rates"] = {LIMBS[L]: {aug.KINDS[q]: round(float(np.mean(kd25[l25 == L] == q)), 4) for q in range(3)} for L in range(4)}
# (4) restoration error on true augmented pairs
aug_pairs = np.flatnonzero(kind_t >= 0)
rest = aug.restore(A25[twin25[aug_pairs]], kind_t[aug_pairs]); err = np.abs(rest - A26[aug_pairs])
edge = [0, 1, 2, 47, 48, 49]
for k in range(3):
    q = kind_t[aug_pairs] == k
    if q.any():
        e = err[q]
        print(f"(4) restore {aug.KINDS[k]}: n {q.sum()} max err all {e.max():.2e} | edge samples max {e[:, edge].max():.2e} | rms all {np.sqrt((e ** 2).mean()):.2e} | per-tile max err q50/q90/q99 {np.quantile(e.max((1, 2)), [0.5, 0.9, 0.99]).round(4)}")
        out[f"restore_{aug.KINDS[k]}_edge_maxerr"] = float(e[:, edge].max()); out[f"restore_{aug.KINDS[k]}_rms"] = float(np.sqrt((e ** 2).mean()))
pos_err = np.sqrt((err ** 2).mean((0, 2)))
print("    restore rms error by sample position (all kinds)\n", pos_err.round(4))
# (5) per-pair models on augmented pairs
SP = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        P = np.zeros((3, 3)); P[list(perm), range(3)] = sg; SP.append(P)


def fits(a, b):
    """relative residual ||b - fit|| / ||b - mean b|| for several model families mapping a -> b"""
    a = a.astype(np.float64); b = b.astype(np.float64); bc = b - b.mean(0); nb = np.linalg.norm(bc) + 1e-12
    r = {}
    X = np.c_[a, np.ones(50)]; M = np.linalg.lstsq(X, b, rcond=None)[0]; r["linear3x3+off"] = np.linalg.norm(b - X @ M) / nb
    s = (a * b).sum() / ((a * a).sum() + 1e-12); r["scale"] = np.linalg.norm(b - s * a) / nb
    sv = (a * b).sum(0) / ((a * a).sum(0) + 1e-12); r["axis-scale"] = np.linalg.norm(b - a * sv) / nb
    r["signed-perm"] = min(np.linalg.norm(b - a @ P) for P in SP) / nb
    U, _, Vt = np.linalg.svd(a.T @ b); R = U @ Vt; r["orthogonal(Procrustes)"] = np.linalg.norm(b - a @ R) / nb
    ac = a - a.mean(0); U, _, Vt = np.linalg.svd(ac.T @ bc); R = U @ Vt; r["orthogonal+offset"] = np.linalg.norm(bc - ac @ R) / nb
    return r, M[:3]


res = {k: [] for k in range(3)}; resw = {k: [] for k in range(3)}; Mfit = {k: [] for k in range(3)}
for i in aug_pairs:
    k = kind_t[i]; a = A26[i]; b = A25[twin25[i]]
    r, M = fits(a, b); res[k].append(r); Mfit[k].append(M)
    # with the true warp applied to a and the true fixed pattern removed from b
    aw = a.astype(np.float64) if k == 1 else np.einsum("pq,qc->pc", aug.WL if k == 0 else aug.WR, a.astype(np.float64))
    bn = b.astype(np.float64) - (0 if k == 0 else (aug.N0 if k == 1 else aug.N1))
    r2, _ = fits(aw, bn); resw[k].append(r2)
print("(5) per-pair model relative residuals (median / q90) on augmented twin pairs:")
summ = {}
for k in range(3):
    if not res[k]:
        continue
    print(f"  kind {aug.KINDS[k]} (n {len(res[k])}):")
    for nm in res[k][0]:
        v = np.array([x[nm] for x in res[k]]); line = f"    {nm:24s} plain {np.median(v):.4f} / {np.quantile(v, 0.9):.4f}"
        summ[f"{aug.KINDS[k]}_{nm}_plain_median"] = round(float(np.median(v)), 5)
        if resw[k]:
            vw = np.array([x[nm] for x in resw[k]]); line += f" | true warp applied, true N removed {np.median(vw):.2e} / {np.quantile(vw, 0.9):.2e}"
            summ[f"{aug.KINDS[k]}_{nm}_warped_denoised_median"] = float(np.median(vw))
        print(line)
    Ms = np.array(Mfit[k]); print("    per-pair 3x3 fit: median M\n", np.median(Ms, 0).round(4), "\n    spread (IQR of entries) max", np.max(np.quantile(Ms, 0.75, 0) - np.quantile(Ms, 0.25, 0)).round(4))
out["model_residuals"] = summ
# transform summary of M0: reflection plane / rotation angle of -M0
Rm = -aug.M0; ang = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
w, V = np.linalg.eig(aug.M0); ax = np.real(V[:, np.argmin(np.abs(w + 1))])
print(f"    M0: det {np.linalg.det(aug.M0):.4f}; -M0 is a proper rotation by {ang:.1f} deg; M0 eigvec for -1 (reflection normal) {ax.round(4)}")
# warp magnitude
for nm, t in (("tauL", aug.TAU_L), ("tauR", aug.TAU_R)):
    d = t - np.arange(50); print(f"    {nm}: shift min {d.min():.3f} max {d.max():.3f} samples; local rate min {np.diff(t).min():.3f} max {np.diff(t).max():.3f}")
dL = aug.TAU_L - np.arange(50); dR = aug.TAU_R - np.arange(50)
out["transform_summary"] = {
    "left_arm L (25.0%)": f"b = -warpL(a): full sign inversion (-I, det -1) + fixed time warp (tau-p in [{dL.min():.2f}, {dL.max():.2f}] samples, local rate {np.diff(aug.TAU_L).min():.2f}-{np.diff(aug.TAU_L).max():.2f}, ends fixed, b[0]==b[1]); no noise, no scaling",
    "right_arm R0 (20.0%)": f"b = a @ M0 + N0: fixed orthogonal M0, det -1 (= -1 x a {ang:.1f} deg rotation), fixed additive pattern N0 (std {aug.N0.std(0).round(3).tolist()} g, mean {aug.N0.mean(0).round(3).tolist()})",
    "right_arm R1 (20.0%)": f"b = warpR(a) @ P1 + N1: P1 = -(swap x,y) (det +1), fixed time warp (tau-p in [{dR.min():.2f}, {dR.max():.2f}]), fixed additive pattern N1 (std {aug.N1.std(0).round(3).tolist()} g)",
    "legs": "never augmented", "scale": "none (all singular values exactly 1)",
    "random_per_row": "none: every parameter is identical for all rows of a kind; only which rows get augmented is random (exact counts: round(rate*n) per subject-limb)"}
for k_, v_ in out["transform_summary"].items():
    print(f"    {k_}: {v_}")
fp26a = np.zeros(len(A26), bool)
for L in range(4):
    j = np.flatnonzero(l26 == L); fp26a[j] = aug.detect(A26[j], limb=LIMBS[L])[0] >= 0
print("    false positives on clean 2026 tiles (limb-aware):", int(fp26a.sum()), "of", len(A26))
out["fp_clean2026_limb_aware"] = int(fp26a.sum())
# cleaned 2025 array: restore every detected row, then overwrite rows that are twins of our tiles with the exact clean tile
acc_r = A25.copy()
for L in range(4):
    j = np.flatnonzero(l25 == L); acc_r[j] = aug.restore(A25[j], kd25[j])
acc_c = acc_r.copy(); acc_c[twin25] = A26
np.savez(os.path.join(D, "w25_restored.npz"), acc_restored=acc_r, acc_clean=acc_c, kind25=kd25, twin25=twin25, id=z["id"], sbj=s25, limb=l25)
print("    saved w25_restored.npz (acc_restored, acc_clean = restored + twin rows replaced by our tiles, kind25, twin25)")
# (6) replaceable fraction
n_aug = int(np.sum(kd25 >= 0)); n_aug_tw = len(np.unique(twin25[aug_pairs]))
print(f"(6) augmented 2025 rows {n_aug}; of these twins of our 2026 tiles {n_aug_tw} -> replaceable fraction {n_aug_tw / n_aug:.4f}; invertible by restore(): {n_aug}/{n_aug} (R0 exact, L/R1 edge-exact)")
out["replaceable_fraction"] = n_aug_tw / n_aug; out["n_aug_rows"] = n_aug; out["n_aug_rows_with_twin"] = n_aug_tw
np.savez(os.path.join(D, "validate.npz"), twin25=twin25, kind_t=kind_t, kind25=kd25, score25=sc25)
json.dump(out, open(os.path.join(D, "validate_summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "model_residuals"}, indent=1))
