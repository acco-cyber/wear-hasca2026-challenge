"""Count-prior lab on a late fusion of fits: every variant predicts the tiles per (subject, exercise), nested by subject
fold on the training subjects; scored by count error and by OOF macro F1 after Sinkhorn on the fused P (pre-refiner).
  python count_lab.py --parts P1,P2,... [--variants all|name,name] [--write name]"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_local as V
from v4_local import H, macro_f1, N_CLS, TRAIN_SETS, FOLDS, profile_features, ridge_cv, count_targets, finish_targets, log, W
from v4_combine import load_part


def ridge_fit(Xtr, ttr, Xte, alphas=(1, 3, 10, 30, 100, 300, 1000)):
    reg = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    return np.clip(ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd, alphas), 70, 135)


def gbm_fit(Xtr, ttr, Xte, seed=0):
    import lightgbm as lgb
    reg = (ttr > 55) & (ttr < 150)
    b = lgb.train(dict(objective="regression_l1", learning_rate=0.03, num_leaves=4, min_data_in_leaf=12, feature_fraction=0.6,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=seed, num_threads=2), lgb.Dataset(Xtr[reg], ttr[reg]), 300)
    return np.clip(b.predict(Xte), 70, 135)


def nested(X, true, kf, Xt, fitter):
    cnt = np.zeros(len(true))
    for f in range(FOLDS):
        cnt[kf == f] = fitter(X[kf != f], true[kf != f], X[kf == f])
    return cnt, fitter(X, true, Xt)


def subj_feats(X, key, cols):
    """per-subject context: the subject's mean / std of selected profile columns, broadcast to its 18 rows"""
    out = np.zeros((len(X), 2 * len(cols)))
    for s in np.unique(key[:, 0]):
        m = key[:, 0] == s; out[m] = np.r_[X[m][:, cols].mean(0), X[m][:, cols].std(0)]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", required=True); ap.add_argument("--variants", default="all")
    ap.add_argument("--write", default=""); ap.add_argument("--tag", default="cl")
    a = ap.parse_args()
    fits = [load_part(s) for s in a.parts.split(",")]
    full = [f for f in fits if not f.get("q_only")]; F = full[0]
    y, sbj, fold, tsbj = F["oof_y"].astype(int), F["oof_sbj"].astype(int), F["oof_fold"].astype(int), F["test_sbj"].astype(int)
    fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(np.mean([np.log(np.clip(M, 1e-9, None)) for M in Ms], 0)))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    Bo = H.lsm(np.mean([f["B2_OOF"].astype(np.float64) for f in full], 0)); Bt = H.lsm(np.mean([f["B2_TEST"].astype(np.float64) for f in full], 0))
    Bpo, Bpt = np.exp(Bo), np.exp(Bt)
    X0, key = profile_features([Po, Bpo], sbj, TRAIN_SETS); X0t, kt = profile_features([Pt, Bpt], tsbj, {})
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    kf = np.array([fold_of[int(s)] for s, _ in key])
    OH, OHt = np.eye(N_CLS - 1)[key[:, 1] - 1], np.eye(N_CLS - 1)[kt[:, 1] - 1]
    # each fit's own profile (for per-fit regressions)
    PF = [profile_features([f["P_o"], np.exp(f["B2_OOF"].astype(np.float64)) if "B2_OOF" in f else Bpo], sbj, TRAIN_SETS)[0] for f in fits]
    PFt = [profile_features([f["P_t"], np.exp(f["B2_TEST"].astype(np.float64)) if "B2_TEST" in f else Bpt], tsbj, {})[0] for f in fits]
    reg = (true > 55) & (true < 150)
    cls_mean = lambda tr_mask: np.array([true[tr_mask & (key[:, 1] == c) & reg].mean() for c in range(1, N_CLS)])

    def V_base():
        return nested(np.c_[X0, OH], true, kf, np.c_[X0t, OHt], ridge_fit)

    def V_noh():
        return nested(X0, true, kf, X0t, ridge_fit)

    def V_perfit_avg():                                    # average of per-fit regressions (+ the joint one)
        c, t = V_base(); cs, ts = [c], [t]
        for Xf, Xft in zip(PF, PFt):
            c_, t_ = nested(np.c_[Xf, OH], true, kf, np.c_[Xft, OHt], ridge_fit); cs.append(c_); ts.append(t_)
        return np.mean(cs, 0), np.mean(ts, 0)

    def V_subj():                                          # + subject context of the margin columns
        cols = list(range(0, 9)) + [9, 10]
        return nested(np.c_[X0, OH, subj_feats(X0, key, cols)], true, kf, np.c_[X0t, OHt, subj_feats(X0t, kt, cols)], ridge_fit)

    def V_resid():                                         # predict the deviation of the subject from its own mean prediction
        c, t = V_base()
        def centre(cnt, k_):
            out = cnt.copy()
            for s in np.unique(k_[:, 0]):
                m = k_[:, 0] == s; out[m] = cnt[m] - cnt[m].mean()
            return out
        return c, t

    def V_gbm():
        return nested(np.c_[X0, OH], true, kf, np.c_[X0t, OHt], gbm_fit)

    def V_ridge_gbm():
        c1, t1 = V_base(); c2, t2 = V_gbm(); return 0.5 * (c1 + c2), 0.5 * (t1 + t2)

    def V_shrink():                                        # shrink toward the class mean, factor chosen in inner folds
        c, t = V_base(); out = np.zeros_like(c); best_all = []
        for f in range(FOLDS):
            tr = kf != f; cm = cls_mean(tr)[key[:, 1] - 1]
            # inner nested predictions for the training folds come from the outer nested c (no label of fold f used)
            errs = {lam: np.abs((1 - lam) * c[tr & reg] + lam * cm[tr & reg] - true[tr & reg]).mean() for lam in (0, 0.1, 0.2, 0.3, 0.4, 0.5)}
            lam = min(errs, key=errs.get); best_all.append(lam); out[kf == f] = (1 - lam) * c[kf == f] + lam * cm[kf == f]
        lam = float(np.median(best_all)); cmt = cls_mean(np.ones(len(true), bool))[kt[:, 1] - 1]
        log(f"  shrink lambdas per fold {best_all}")
        return out, (1 - lam) * t + lam * cmt

    def V_log():
        def fit(Xtr, ttr, Xte):
            reg_ = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg_].mean(0), Xtr[reg_].std(0) + 1e-6
            return np.clip(np.exp(ridge_cv((Xtr[reg_] - mu) / sd, np.log(ttr[reg_]), (Xte - mu) / sd)), 70, 135)
        return nested(np.c_[X0, OH], true, kf, np.c_[X0t, OHt], fit)

    def V_combo():                                         # per-fit average + gbm
        c1, t1 = V_perfit_avg(); c2, t2 = V_gbm(); return 0.5 * (c1 + c2), 0.5 * (t1 + t2)

    VARS = dict(base=V_base, no_onehot=V_noh, perfit_avg=V_perfit_avg, subj=V_subj, gbm=V_gbm, ridge_gbm=V_ridge_gbm,
                shrink=V_shrink, log=V_log, combo=V_combo)
    names = list(VARS) if a.variants == "all" else a.variants.split(",")
    res = {}
    for nm in names:
        cnt, ct = VARS[nm]()
        tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
        Qo = finish_targets(Po, sbj, tg); lab = Qo.argmax(1)
        pf = [macro_f1(y[fold == f_], lab[fold == f_]) for f_ in range(FOLDS)]
        res[nm] = (macro_f1(y, lab), pf, cnt, ct, tg, tgt)
        log(f"{nm:12s} count MAE {np.abs(cnt - true)[reg].mean():.2f} (all {np.abs(cnt - true).mean():.2f}) | OOF pre-refiner {res[nm][0]:.4f} | per fold "
            + " ".join(f"{x:.4f}" for x in pf) + f" | test means {dict((int(s), round(float(ct[kt[:, 0] == s].mean()), 1)) for s in np.unique(tsbj))}")
    if a.write:
        f1, pf, cnt, ct, tg, tgt = res[a.write]
        np.savez(os.path.join(W, "subs", f"counts_{a.tag}_{a.write}.npz"), cnt=cnt, ct=ct, key=key, kt=kt)
        log(f"saved counts of {a.write}")


if __name__ == "__main__":
    main()
