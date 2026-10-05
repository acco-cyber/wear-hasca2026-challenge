"""Count-prior variants for the fused v4 decode: features, regressors, nested evaluation helpers."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
from v4_local import macro_f1, N_CLS, TRAIN_SETS, FOLDS, profile_features, fit_counts, ridge_cv, count_targets, finish_targets

HERE = os.path.dirname(os.path.abspath(__file__))
LO, HI = 70, 135


def load():
    Z = np.load(os.path.join(HERE, "cache.npz")); return {k: Z[k] for k in Z.files}


def subject_feats(mats, sbj, sets, key):
    """per-subject scalars repeated on each (subject, exercise) row: null share (soft / argmax), soft non-null budget per
    exercise; plus the row's soft / argmax count relative to its subject's mean over the 18 exercises"""
    M = mats[0]; out = []
    for s, c in key:
        ii = sbj == s; ns = sets.get(int(s), 1); n = ii.sum()
        soft = M[ii][:, 1:].sum(0) / ns; am = np.bincount(M[ii].argmax(1), minlength=N_CLS)[1:] / ns
        out.append([M[ii, 0].mean(), np.mean(M[ii].argmax(1) == 0), (n - M[ii, 0].sum()) / ns / 18.0,
                    soft[c - 1] / soft.mean(), am[c - 1] / max(am.mean(), 1e-6), soft[c - 1] - soft.mean(), am[c - 1] - am.mean()])
    return np.array(out, np.float64)


RK_EXT = [160, 180, 200, 230]


def ext_feats(mats, sbj, sets):
    """margin profile beyond rank 140 and hinges on the argmax / positive-margin counts (lets a linear model say 'more
    than the usual ~100 tiles' when the evidence is strong)"""
    X = []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); ns = sets.get(int(s), 1)
        for c in range(1, N_CLS):
            f = []
            for M in mats:
                q = M[ii][:, c]; oth = np.delete(M[ii], c, 1).max(1)
                sc = np.sort(np.log(q + 1e-6) - np.log(oth + 1e-6))[::-1]
                f += [sc[min(int(r * ns), len(sc) - 1)] for r in RK_EXT]
                am = (M[ii].argmax(1) == c).sum() / ns; pos = (sc > 0).sum() / ns
                f += [max(am - 120, 0), max(am - 150, 0), max(pos - 120, 0), max(pos - 150, 0), max(70 - am, 0), max(70 - pos, 0)]
            X.append(f)
    return np.array(X, np.float64)


def build(D, mats_o, mats_t, onehot=True, subj=False, center=False, ext=False):
    sbj, tsbj = D["sbj"], D["tsbj"]
    X, key = profile_features(mats_o, sbj, TRAIN_SETS); Xt, kt = profile_features(mats_t, tsbj, {})
    if ext:
        X = np.concatenate([X, ext_feats(mats_o, sbj, TRAIN_SETS)], 1); Xt = np.concatenate([Xt, ext_feats(mats_t, tsbj, {})], 1)
    if subj:
        X = np.concatenate([X, subject_feats(mats_o, sbj, TRAIN_SETS, key)], 1); Xt = np.concatenate([Xt, subject_feats(mats_t, tsbj, {}, kt)], 1)
    if center:                                    # append subject-centred copies of every feature
        def cen(A, k):
            C = A.copy()
            for s in np.unique(k[:, 0]):
                m = k[:, 0] == s; C[m] -= C[m].mean(0)
            return C
        X = np.concatenate([X, cen(X, key)], 1); Xt = np.concatenate([Xt, cen(Xt, kt)], 1)
    if onehot:
        X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1); Xt = np.concatenate([Xt, np.eye(N_CLS - 1)[kt[:, 1] - 1]], 1)
    y, sbj = D["y"], D["sbj"]
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    fold_of = {int(s): int(f_) for s, f_ in zip(D["sbj"], D["fold"])}
    kf = np.array([fold_of[int(s)] for s, _ in key])
    return X, key, Xt, kt, true, kf


# ------------------------------------------------------------------ regressors: fit on (Xtr, ttr, ktr), predict Xte (kte)
def _std(Xtr, reg):
    mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6; return mu, sd


def m_ridge(Xtr, ttr, ktr, Xte, kte):
    return fit_counts(Xtr, ttr, Xte)


def m_ridge_log(Xtr, ttr, ktr, Xte, kte):
    reg = (ttr > 55) & (ttr < 150); mu, sd = _std(Xtr, reg)
    return np.clip(np.exp(ridge_cv((Xtr[reg] - mu) / sd, np.log(ttr[reg]), (Xte - mu) / sd)), LO, HI)


def _centre_t(t, k):
    out = t.copy()
    for s in np.unique(k[:, 0]):
        m = k[:, 0] == s; out[m] -= t[m].mean()
    return out


def m_ridge_cen(Xtr, ttr, ktr, Xte, kte):
    """subject mean from the plain ridge, deviation from the subject mean from a ridge on subject-centred features/target"""
    base = fit_counts(Xtr, ttr, Xte)
    reg_s = np.array([((ttr[ktr[:, 0] == s] > 55) & (ttr[ktr[:, 0] == s] < 150)).all() for s in ktr[:, 0]])  # subjects w/o irregular keys
    Xc_tr, Xc_te = _centre_t_mat(Xtr, ktr), _centre_t_mat(Xte, kte)
    tc = _centre_t(ttr, ktr); reg = reg_s
    mu, sd = Xc_tr[reg].mean(0), Xc_tr[reg].std(0) + 1e-6
    dev = ridge_cv((Xc_tr[reg] - mu) / sd, tc[reg], (Xc_te - mu) / sd)
    out = np.zeros(len(Xte))
    for s in np.unique(kte[:, 0]):
        m = kte[:, 0] == s; out[m] = base[m].mean() + dev[m] - dev[m].mean()
    return np.clip(out, LO, HI)


def _centre_t_mat(A, k):
    C = A.copy()
    for s in np.unique(k[:, 0]):
        m = k[:, 0] == s; C[m] -= C[m].mean(0)
    return C


LGB_P = dict(objective="l1", learning_rate=0.03, num_leaves=5, min_data_in_leaf=12, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=0)


def m_lgb(Xtr, ttr, ktr, Xte, kte, rounds=400):
    import lightgbm as lgb
    reg = (ttr > 55) & (ttr < 150); preds = []
    for sd in range(3):
        p = dict(LGB_P, seed=sd)
        preds.append(lgb.train(p, lgb.Dataset(Xtr[reg], ttr[reg]), rounds).predict(Xte))
    return np.clip(np.mean(preds, 0), LO, HI)


def m_avg(Xtr, ttr, ktr, Xte, kte):
    return 0.5 * m_ridge(Xtr, ttr, ktr, Xte, kte) + 0.5 * m_lgb(Xtr, ttr, ktr, Xte, kte)


# wide variants: trained on every key with t > 30 (only the absent exercise is left out), wider output range
WLO, WHI, WT = 40, 230, 30


def m_ridge_w(Xtr, ttr, ktr, Xte, kte):
    reg = ttr > WT; mu, sd = _std(Xtr, reg)
    return np.clip(ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd), WLO, WHI)


def m_ridge_log_w(Xtr, ttr, ktr, Xte, kte):
    reg = ttr > WT; mu, sd = _std(Xtr, reg)
    return np.clip(np.exp(ridge_cv((Xtr[reg] - mu) / sd, np.log(ttr[reg]), (Xte - mu) / sd)), WLO, WHI)


def m_lgb_w(Xtr, ttr, ktr, Xte, kte, rounds=400):
    import lightgbm as lgb
    reg = ttr > WT; preds = []
    for sd in range(3):
        preds.append(lgb.train(dict(LGB_P, seed=sd), lgb.Dataset(Xtr[reg], ttr[reg]), rounds).predict(Xte))
    return np.clip(np.mean(preds, 0), WLO, WHI)


def m_avg_w(Xtr, ttr, ktr, Xte, kte):
    return 0.5 * m_ridge_log_w(Xtr, ttr, ktr, Xte, kte) + 0.5 * m_lgb_w(Xtr, ttr, ktr, Xte, kte)


def m_avg_log(Xtr, ttr, ktr, Xte, kte):
    return 0.5 * m_ridge_log(Xtr, ttr, ktr, Xte, kte) + 0.5 * m_lgb(Xtr, ttr, ktr, Xte, kte)


MODELS = dict(ridge=m_ridge, ridge_log=m_ridge_log, ridge_cen=m_ridge_cen, lgb=m_lgb, avg=m_avg, avg_log=m_avg_log,
              ridge_w=m_ridge_w, ridge_log_w=m_ridge_log_w, lgb_w=m_lgb_w, avg_w=m_avg_w)


def class_means(ttr, ktr, kte):
    reg = (ttr > 55) & (ttr < 150); cm = np.array([ttr[reg & (ktr[:, 1] == c)].mean() for c in range(1, N_CLS)])
    return cm[kte[:, 1] - 1]


def cv_predict(model, X, true, key, kf, folds_eval, folds_pool):
    """for every fold j in folds_eval: fit on folds_pool minus j, predict j"""
    out = np.full(len(true), np.nan)
    for j in folds_eval:
        tr = np.isin(kf, [f for f in folds_pool if f != j]); te = kf == j
        out[te] = model(X[tr], true[tr], key[tr], X[te], key[te])
    return out


def f1_from_counts(D, P, key, cnt, subjects=None):
    sbj, y = D["sbj"], D["y"]
    if subjects is None:
        subjects = np.unique(key[:, 0])
    m_k = np.isin(key[:, 0], subjects); m_t = np.isin(sbj, subjects)
    tg = count_targets(sbj[m_t], TRAIN_SETS, key[m_k], cnt[m_k])
    lab = finish_targets(P[m_t], sbj[m_t], tg).argmax(1)
    return macro_f1(y[m_t], lab), lab


def centred_err(cnt, true, key):
    e = []
    for s in np.unique(key[:, 0]):
        m = key[:, 0] == s; e.append(np.abs((cnt[m] - cnt[m].mean()) - (true[m] - true[m].mean())))
    return np.concatenate(e).mean()
