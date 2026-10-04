"""Offline study of the learned count prior, from the profile features cached by our fork (dec_cache.npz):
X (22 subjects x 18 classes rows, features of the margin profile), true counts. Nested by subject fold.
Questions: how much of the count error is a per-subject bias? do subject-level features (means over the subject's
18 classes) or class identity reduce the error?  python count_lab.py [keep3 dir]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from good_local import ridge_cv
W = r"E:\Claude code\wear"; GK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(W, "work", "good", "keep3")
dc = np.load(os.path.join(GK, "dec_cache.npz")); st = np.load(os.path.join(GK, "stage.npz"))
sbj, fold = st["oof_sbj"].astype(int), st["oof_fold"].astype(int); fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
subs = np.unique(sbj); key = np.array([(s, c) for s in subs for c in range(1, 19)]); kf = np.array([fold_of[int(s)] for s, _ in key])


def fit(Xtr, ttr, Xte, clip=(70, 135)):
    reg = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    return np.clip(ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd), *clip)


def nested(X, true):
    cnt = np.zeros(len(true))
    for f in range(5):
        cnt[kf == f] = fit(X[kf != f], true[kf != f], X[kf == f])
    return cnt


def report(nm, cnt, true):
    ok = (true > 55) & (true < 150); e = cnt - true
    sb = np.array([e[(key[:, 0] == s) & ok].mean() for s in subs])          # per-subject bias
    within = e.copy()
    for i, s in enumerate(subs):
        within[key[:, 0] == s] -= sb[i]
    print(f"{nm:44s} MAE {np.abs(e[ok]).mean():.2f} | subject bias: mean |bias| {np.abs(sb).mean():.2f} (max {np.abs(sb).max():.1f}) | MAE after removing the true subject bias {np.abs(within[ok]).mean():.2f}")


for k in (0, 1):
    X, true = dc[f"stage_B_{k}_X"], dc[f"stage_B_{k}_true"]; assert len(X) == len(key)
    print(f"\n== pass {k + 1}: X {X.shape}")
    report("fixed 97", np.full(len(true), 97.0), true)
    report("their ridge on margin profiles", nested(X, true), true)
    # subject-level features: mean and median over the subject's 18 classes of every feature
    G = np.zeros_like(X); Gm = np.zeros_like(X)
    for s in subs:
        m = key[:, 0] == s; G[m] = X[m].mean(0); Gm[m] = np.median(X[m], 0)
    report("+ subject means of the features", nested(np.concatenate([X, G], 1), true), true)
    report("+ subject means and medians", nested(np.concatenate([X, G, Gm], 1), true), true)
    report("deviation from subject mean + subject mean", nested(np.concatenate([X - G, G], 1), true), true)
    C = np.eye(18)[key[:, 1] - 1]
    report("+ class one-hot", nested(np.concatenate([X, C], 1), true), true)
    report("+ subject means + class one-hot", nested(np.concatenate([X, G, C], 1), true), true)
    # two-step: predict the subject mean count from subject-level features, then the class deviation
    ok = (true > 55) & (true < 150)
    sm_true = np.array([true[(key[:, 0] == s) & ok].mean() for s in subs]); Xs = np.array([X[key[:, 0] == s].mean(0) for s in subs]); sf = np.array([fold_of[int(s)] for s in subs])
    sm_pred = np.zeros(len(subs))
    for f in range(5):
        mu, sd = Xs[sf != f].mean(0), Xs[sf != f].std(0) + 1e-6
        sm_pred[sf == f] = ridge_cv((Xs[sf != f] - mu) / sd, sm_true[sf != f], (Xs[sf == f] - mu) / sd, alphas=(3, 10, 30, 100, 300, 1000, 3000))
    print(f"   subject mean count: true range {sm_true.min():.0f}-{sm_true.max():.0f} (std {sm_true.std():.1f}); predicted from subject-level features MAE {np.abs(sm_pred - sm_true).mean():.2f} (constant: {np.abs(sm_true.mean() - sm_true).mean():.2f})")
