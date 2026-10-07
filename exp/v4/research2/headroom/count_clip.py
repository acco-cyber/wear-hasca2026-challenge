"""Label-free test-side check of the count prior: how often does the ridge count regressor's RAW prediction leave the
[70, 135] clip on the OOF keys (nested) vs the test keys, and how did the clipped OOF keys compare with the truth?
Profile features from K7's kernel graph P (PB_*) and stage-B blend, exercise identity on (as in v4_local --onehot)."""
import numpy as np
from hlib import *
from v4_local import ridge_cv

D = setup(); F = D["F"]; y, sbj = D["y"], D["sbj"]; tsbj = F["test_sbj"].astype(np.int64)
X, key = profile_features([F["PB_OOF"].astype(np.float64), np.exp(F["B2_OOF"].astype(np.float64))], sbj, TRAIN_SETS)
Xt, kt = profile_features([F["PB_TEST"].astype(np.float64), np.exp(F["B2_TEST"].astype(np.float64))], tsbj, {})
X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1); Xt = np.concatenate([Xt, np.eye(N_CLS - 1)[kt[:, 1] - 1]], 1)
true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)


def raw_fit(Xtr, ttr, Xte):
    reg = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    return ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd)


kf = np.array([D["fold_of"][int(s)] for s, _ in key]); raw = np.zeros(len(true))
for f_ in range(FOLDS):
    raw[kf == f_] = raw_fit(X[kf != f_], true[kf != f_], X[kf == f_])
rt = raw_fit(X, true, Xt)
hi, lo = raw > 135, raw < 70
log(f"OOF keys {len(raw)}: raw>135 {hi.sum()} (true there: {np.round(true[hi], 0).tolist()}), raw<70 {lo.sum()} (true there: {np.round(true[lo], 0).tolist()})")
log(f"OOF truly irregular keys (true>135 or <70): {np.sum((true > 135) | (true < 70))}; their raw preds {np.round(raw[(true > 135) | (true < 70)], 0).tolist()}")
log(f"TEST keys {len(rt)}: raw>135 {np.sum(rt > 135)}, raw<70 {np.sum(rt < 70)}")
for s in np.unique(tsbj):
    m = kt[:, 0] == s; n = int((tsbj == s).sum())
    log(f"  test sbj {s}: n={n}, raw count pred mean {rt[m].mean():.1f} min {rt[m].min():.0f} max {rt[m].max():.0f}; implied null {n - np.clip(rt[m], 70, 135).sum():.0f} ({(n - np.clip(rt[m], 70, 135).sum()) / n:.2f})")
log(f"OOF null fraction range {min((y[sbj == s] == 0).mean() for s in np.unique(sbj)):.2f}-{max((y[sbj == s] == 0).mean() for s in np.unique(sbj)):.2f}")
