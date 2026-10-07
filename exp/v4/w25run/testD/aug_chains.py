"""Realistic simulation chains (mode 'aug'): the 2025 hosts' augmentation hits only the arms in the real data
(exact-twin share: right_arm ~0.59, left_arm ~0.75, legs 1.0). Measured on twins: right_arm rows = x @ R (improper rotation,
pooled Procrustes on confident twins) + white noise sd ~0.104 g; left_arm rows ~ -x (best signed permutation by energy
distance; approximate). Augmented seconds are almost never consecutive (same-limb links of our tiles: NN 0.026 vs 0.165
independent) -> Markov pattern with P(aug|aug)=0.05. Here the training subjects' arm chains are rebuilt on tiles augmented
that way (own tiles at augmented seconds = unanchored, exactly like the test), features + nested chain LightGBM as before.
  python aug_chains.py  -> cache/chainaug_s{s}.npz (rows, true_loc, succ/conf per limb, aug (4,n))"""
import os, sys, time, glob, csv
import numpy as np
from joblib import Parallel, delayed
from tdlib import *

T0 = time.time()
def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


S = load_stage(K7); fold_of = {int(s): int(f) for s, f in zip(S["oof_sbj"], S["oof_fold"])}
P_NN = 0.05


def train_fold_models():
    import lightgbm as lgb
    need = [(f, os.path.join(MODELS, f"chain_f{f}.txt")) for f in range(5)]
    if all(os.path.exists(p) for _, p in need):
        return
    subs = list(range(22)); Xs, ys, ss = [], [], []
    for s in subs:
        z = np.load(os.path.join(FEAS, "cache", f"feat_s{s}.npz"))
        for L in range(4):
            X = z[f"X{L}"]; Xs.append(X); ys.append(z[f"y{L}"]); ss.append(np.full(len(X), s))
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.int8); sb = np.concatenate(ss); fo = np.array([fold_of[s] for s in subs])[sb]
    del Xs
    rr, rc = X[:, FEATS.index("rank_row")], X[:, FEATS.index("rank_col")]
    rng = np.random.default_rng(0); train_mask = (y == 1) | (rr < 4) | (rc < 4) | (rng.random(len(y)) < 0.05)
    for f, p in need:
        if os.path.exists(p):
            continue
        tr = (fo != f) & train_mask
        m = lgb.train(P_CHAIN, lgb.Dataset(X[tr], y[tr]), ROUNDS_CHAIN); m.save_model(p)
        te = fo == f; pr = m.predict(X[te], raw_score=True); ref = np.load(os.path.join(FEAS, "cache", f"chain_lgb_pred_f{f}.npy"))
        log(f"chain fold model {f}: max|diff| vs cached nested predictions {np.abs(pr - ref).max():.5f}")


def test_params():
    A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
    z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64)
    m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
    st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); sens = st["sensor_test"]        # pipeline order == data order (asserted in chains.py)
    TI = np.load(os.path.join(CACHE, "test_inputs.npz")); exact = TI["exact_d"]
    pi_aug = np.array([1 - exact[sens == L].mean() for L in range(4)])
    k = np.flatnonzero(~exact & (corr > 0.99) & (margin > 0.05) & (sens == 0))
    Xs = np.concatenate([A26[i] for i in k]); Bs = np.concatenate([A25[twin[i]] for i in k])
    U, _, Vt = np.linalg.svd(Xs.T @ Bs); R = U @ Vt
    sd = float(np.median([(A25[twin[i]] - A26[i] @ R).std() for i in k]))
    return pi_aug, R, sd


def augment_subject(s, pi_aug, R, sd):
    t0 = time.time(); ii, T = subject_tiles(S, s); n = len(ii)
    pos = np.full(len(S["oof_y"]), -1); pos[ii] = np.arange(n)
    ts = S["true_succ"][ii]; tl = np.where(ts >= 0, pos[np.maximum(ts, 0)], -1)
    rec = S["oof_rec"][ii]; first = np.r_[True, rec[1:] != rec[:-1]]
    rng = np.random.default_rng(5000 + s)
    aug = np.zeros((4, n), bool)
    for L in (0, 3):
        p = pi_aug[L]; q = p * (1 - P_NN) / (1 - p); u = rng.random(n); st_ = False
        for t in range(n):
            if first[t]:
                st_ = u[t] < p
            else:
                st_ = u[t] < (P_NN if st_ else q)
            aug[L, t] = st_
    out = dict(rows=ii, true_loc=tl, aug=aug)
    z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); assert (z["rows"] == ii).all()
    msg = []
    for L in range(4):
        if L in (1, 2):
            out[f"succ{L}"], out[f"conf{L}"] = z[f"succ{L}"], z[f"conf{L}"]
        else:
            X = T[L].astype(np.float64).copy(); k = aug[L]
            if L == 0:
                X[k] = X[k] @ R + rng.normal(0, sd, X[k].shape)
            else:
                X[k] = -X[k]
            _, su, cf = score_chain_limb(X.astype(np.float32), L, os.path.join(MODELS, f"chain_f{fold_of[s]}.txt"))
            out[f"succ{L}"], out[f"conf{L}"] = su, cf
        h = tl >= 0
        msg.append(f"L{L} aug {aug[L].mean():.3f} exact {np.mean(out[f'succ{L}'][h] == tl[h]):.3f} conf {np.mean(out[f'conf{L}']):.3f}")
    np.savez(os.path.join(CACHE, f"chainaug_s{s}.npz"), **out)
    return f"sbj {s} n={n}: " + " | ".join(msg) + f" [{time.time() - t0:.0f}s]"


def deaug_subject(s, pi_aug, q2, A2, N2):
    """variant 'deaug' (matches the test after deaug.py): left_arm fully restored (cached clean chain, all anchored);
    right_arm: augmented seconds (same Markov pattern) split into type i (restored exactly -> clean, anchored) and type ii
    (share q2; transform unknown -> proxy x @ A2 + N2 from typeii_probe, unanchored)."""
    t0 = time.time(); ii, T = subject_tiles(S, s); n = len(ii)
    pos = np.full(len(S["oof_y"]), -1); pos[ii] = np.arange(n)
    ts = S["true_succ"][ii]; tl = np.where(ts >= 0, pos[np.maximum(ts, 0)], -1)
    rec = S["oof_rec"][ii]; first = np.r_[True, rec[1:] != rec[:-1]]
    rng = np.random.default_rng(5000 + s)
    aug = np.zeros((4, n), bool); p = pi_aug[0]; q = p * (1 - P_NN) / (1 - p); u = rng.random(n); st_ = False
    for t in range(n):
        st_ = (u[t] < p) if first[t] else (u[t] < (P_NN if st_ else q))
        aug[0, t] = st_
    aug[0] &= rng.random(n) < q2                                       # type ii only
    out = dict(rows=ii, true_loc=tl, aug=aug)
    z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); assert (z["rows"] == ii).all()
    for L in (1, 2, 3):
        out[f"succ{L}"], out[f"conf{L}"] = z[f"succ{L}"], z[f"conf{L}"]
    X = T[0].astype(np.float64).copy(); k = aug[0]; X[k] = X[k] @ A2 + N2[None]
    _, out["succ0"], out["conf0"] = score_chain_limb(X.astype(np.float32), 0, os.path.join(MODELS, f"chain_f{fold_of[s]}.txt"))
    h = tl >= 0
    np.savez(os.path.join(CACHE, f"chaindeaug_s{s}.npz"), **out)
    return f"sbj {s} n={n}: L0 type-ii {aug[0].mean():.3f} exact {np.mean(out['succ0'][h] == tl[h]):.3f} conf {np.mean(out['conf0']):.3f} [{time.time() - t0:.0f}s]"


if __name__ == "__main__":
    if "--variant" in sys.argv and sys.argv[sys.argv.index("--variant") + 1] == "deaug":
        pi_aug, R, sd = test_params()
        d = np.load(os.path.join(CACHE, "deaug_inputs.npz")); how = d["how"]
        st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); sens = st["sensor_test"]
        TI = np.load(os.path.join(CACHE, "test_inputs.npz")); exact = TI["exact_d"]
        ne = (sens == 0) & ~exact; q2 = float(np.mean(how[ne] == 0))
        pr = np.load(os.path.join(CACHE, "typeii_probe.npz"))
        log(f"deaug variant: right_arm augmented share {pi_aug[0]:.4f}, type-ii share among augmented {q2:.4f}")
        subs = sorted(range(22), key=lambda s: -(S["oof_sbj"] == s).sum())
        for m in Parallel(n_jobs=3)(delayed(deaug_subject)(s, pi_aug, q2, pr["A"], pr["N"]) for s in subs):
            log(m)
        log("done"); sys.exit(0)
    train_fold_models()
    pi_aug, R, sd = test_params()
    log(f"augmented share per pipeline limb {np.round(pi_aug, 4).tolist()}; right_arm noise sd {sd:.4f}; R det {np.linalg.det(R):.3f}\n{np.round(R, 3)}")
    subs = sorted(range(22), key=lambda s: -(S["oof_sbj"] == s).sum())
    for m in Parallel(n_jobs=3)(delayed(augment_subject)(s, pi_aug, R, sd) for s in subs):
        log(m)
    log("done")
