"""More count-regressor variants on the cached profile features (nested by subject fold); MAE on regular bouts."""
import os, sys
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from good_local import ridge_cv
W = r"E:\Claude code\wear"; GK = os.path.join(W, "work", "good", "keep3")
dc = np.load(os.path.join(GK, "dec_cache.npz")); st = np.load(os.path.join(GK, "stage.npz"))
sbj, fold = st["oof_sbj"].astype(int), st["oof_fold"].astype(int); fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
subs = np.unique(sbj); key = np.array([(s, c) for s in subs for c in range(1, 19)]); kf = np.array([fold_of[int(s)] for s, _ in key])
C = np.eye(18)[key[:, 1] - 1]
FAM = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 2, 12: 2, 13: 3, 14: 3, 15: 4, 16: 5, 17: 5, 18: 6}
F = np.eye(7)[[FAM[int(c)] for c in key[:, 1]]]


def ridge_fit(Xtr, ttr, Xte, alphas=(1, 3, 10, 30, 100, 300, 1000), clip=(70, 135)):
    reg = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    return np.clip(ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd, alphas), *clip)


def gbm_fit(Xtr, ttr, Xte, clip=(70, 135)):
    reg = (ttr > 55) & (ttr < 150)
    b = lgb.train(dict(objective="regression_l1", learning_rate=0.05, num_leaves=4, min_data_in_leaf=10, feature_fraction=0.7, lambda_l2=5.0, verbose=-1, seed=0, num_threads=4),
                  lgb.Dataset(Xtr[reg], ttr[reg]), 150)
    return np.clip(b.predict(Xte), *clip)


def nested(fn, X, true):
    cnt = np.zeros(len(true))
    for f in range(5):
        cnt[kf == f] = fn(X[kf != f], true[kf != f], X[kf == f])
    return cnt


for k in (0, 1):
    X, true = dc[f"stage_B_{k}_X"], dc[f"stage_B_{k}_true"]; ok = (true > 55) & (true < 150)
    mae = lambda c: np.abs(c - true)[ok].mean()
    print(f"\n== pass {k + 1}")
    base = nested(ridge_fit, X, true); oh = nested(ridge_fit, np.concatenate([X, C], 1), true)
    print(f"ridge {mae(base):.2f} | + class one-hot {mae(oh):.2f} | + family one-hot {mae(nested(ridge_fit, np.concatenate([X, F], 1), true)):.2f} "
          f"| + class + family {mae(nested(ridge_fit, np.concatenate([X, C, F], 1), true)):.2f}")
    print(f"one-hot, clip 60-145: {mae(nested(lambda a, b, c: ridge_fit(a, b, c, clip=(60, 145)), np.concatenate([X, C], 1), true)):.2f} | clip 75-130: {mae(nested(lambda a, b, c: ridge_fit(a, b, c, clip=(75, 130)), np.concatenate([X, C], 1), true)):.2f}")
    print(f"one-hot x2 weight (scaled 2): {mae(nested(ridge_fit, np.concatenate([X, 2 * C], 1), true)):.2f} | one-hot + squares of profile: {mae(nested(ridge_fit, np.concatenate([X, X[:, :9] ** 2, C], 1), true)):.2f}")
    g = nested(gbm_fit, np.concatenate([X, key[:, 1:2].astype(float)], 1), true)
    print(f"small GBM (L1) with class id {mae(g):.2f} | mean(ridge one-hot, GBM) {mae((oh + g) / 2):.2f}")
    # class-wise residual correction (target encoding of the residual, nested)
    res = np.zeros(len(true))
    for f in range(5):
        tr, te = kf != f, kf == f; r0 = ridge_fit(X[tr], true[tr], X[tr]); rr = true[tr] - r0; okt = (true[tr] > 55) & (true[tr] < 150)
        cm = np.array([rr[okt & (key[tr, 1] == c)].mean() if (okt & (key[tr, 1] == c)).any() else 0 for c in range(1, 19)])
        res[te] = np.clip(ridge_fit(X[tr], true[tr], X[te]) + 0.7 * cm[key[te, 1] - 1], 70, 135)
    print(f"ridge + 0.7 x class mean residual {mae(res):.2f}")
