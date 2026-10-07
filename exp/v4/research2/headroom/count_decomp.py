"""Count lever, finish-only (no re-decode): which part of the true-count oracle gain is the null count, which the
exercise split; F1 as a function of the count error (shrunk learned error, and true counts + Gaussian noise).
K7 (P of the 2-pass decode, k7_base.npz) and the fused b4wa (re-Sinkhorn of its calibrated Q: the Sinkhorn projection of
D1 Q D2 equals that of the fused P**2, so re-calibrating Qo with other targets == finishing the fused P with them)."""
import numpy as np
from hlib import *

D = setup(); y, sbj, fold = D["y"], D["sbj"], D["fold"]
z = np.load(os.path.join(HERE, "k7_base.npz")); P = z["P"].astype(np.float64)
cnt, key, true = learned_counts(D, P)
TT = true_counts(D)
log(f"K7 learned count error {np.abs(cnt - true).mean():.2f} (regular keys {np.abs(cnt - true)[(true > 55) & (true < 150)].mean():.2f})")
tg_l = targets_from(D, key, cnt)
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64)
tg_f = {int(s): Qo[sbj == s].sum(0) for s in np.unique(sbj)}


def f1_fin(Pm, tg, base_is_Q=False):
    Q = calibrate_targets(Pm, sbj, tg) if base_is_Q else finish_targets(Pm, sbj, tg)
    return macro_f1(y, Q.argmax(1))


def variants(tg_l):
    out = {}
    out["learned"] = tg_l; out["true"] = TT
    out["true null, learned split"] = {s: np.r_[TT[s][0], tg_l[s][1:] / tg_l[s][1:].sum() * (TT[s][1:].sum())] for s in TT}
    out["learned null, true split"] = {s: np.r_[tg_l[s][0], TT[s][1:] / TT[s][1:].sum() * (tg_l[s][1:].sum())] for s in TT}
    for lam in (0.25, 0.5, 0.75):
        out[f"error x{1 - lam:.2f}"] = {s: tg_l[s] + lam * (TT[s] - tg_l[s]) for s in TT}
    return out


for nm, base, tg0, isQ in (("K7", P, tg_l, False), ("b4wa", Qo, tg_f, True)):
    err = np.mean([np.abs(tg0[s][1:] - TT[s][1:]).mean() for s in TT]); nerr = np.mean([abs(tg0[s][0] - TT[s][0]) for s in TT])
    log(f"{nm}: target error per exercise {err:.2f} tiles, null {nerr:.1f} tiles")
    for v, tg in variants(tg0).items():
        log(f"  {nm} finish with {v:28s}: {f1_fin(base, tg, isQ):.4f}")
    # true counts + noise: how small must the count error be?
    rng = np.random.default_rng(0)
    for sd in (2, 4, 6, 8):
        f = []
        for rep in range(3):
            tg = {}
            for s in TT:
                t = TT[s].copy(); t[1:] = np.clip(t[1:] + rng.normal(0, sd, 18) * (t[1:] > 0), 0, None)
                t[0] = max(TT[s].sum() - t[1:].sum(), 0.05 * TT[s].sum()); tg[s] = t
            f.append(f1_fin(base, tg, isQ))
        log(f"  {nm} true exercise counts + N(0,{sd}) noise (null = rest): {np.mean(f):.4f} +- {np.std(f):.4f}")
    # per subject gain of the true counts
    Qt = calibrate_targets(base, sbj, TT) if isQ else finish_targets(base, sbj, TT)
    Ql = calibrate_targets(base, sbj, tg0) if isQ else finish_targets(base, sbj, tg0)
    log(f"  {nm} per-subject F1 learned -> true: " + " ".join(
        f"{s}:{macro_f1(y[sbj == s], Ql[sbj == s].argmax(1)):.3f}->{macro_f1(y[sbj == s], Qt[sbj == s].argmax(1)):.3f}" for s in np.unique(sbj)))
    # error per subject
    log(f"  {nm} per-subject exercise-count MAE / null error: " + " ".join(
        f"{s}:{np.abs(tg0[s][1:] - TT[s][1:]).mean():.1f}/{tg0[s][0] - TT[s][0]:+.0f}" for s in TT))
