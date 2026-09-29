import numpy as np
from common import *

d = load_oof()
y, rec, st, sbj, fold, sen = d["y"], d["rec"], d["start"], d["sbj"], d["fold"], d["sensor"]
print("keys", list(d))
print("n", len(y), "subjects", np.unique(sbj), "recs", np.unique(rec), "folds", np.unique(fold, return_counts=True))
print("sensor", np.unique(sen, return_counts=True))
for s in np.unique(sbj):
    m = sbj == s
    print("sbj", s, "n", m.sum(), "recs", np.unique(rec[m]), "fold", np.unique(fold[m]), "null", round(np.mean(y[m] == 0), 3),
          "sensor", np.unique(sen[m], return_counts=True)[1])
# duplicates in time within rec?
for r in np.unique(rec)[:3]:
    ii = np.flatnonzero(rec == r); t = st[ii]
    print("rec", r, "n", len(ii), "unique starts", len(np.unique(t)), "t range", t.min() // 50, t.max() // 50, "step", np.median(np.diff(np.sort(t))))
pred = finish(d["P"], dict(sbj=sbj, sets=TRAIN_SETS))
print("baseline", round(macro_f1(y, pred), 4), per_fold(y, pred, fold))
bid = true_bouts(y, rec, st)
maj = pred.copy()
for b in np.unique(bid):
    ii = np.flatnonzero(bid == b); maj[ii] = np.bincount(pred[ii], minlength=N_CLS).argmax()
print("oracle bout-majority", round(macro_f1(y, maj), 4), per_fold(y, maj, fold))
# links
succ = d["succ"]; m = succ >= 0
print("links", m.mean(), "same label", np.mean(y[m] == y[succ[m]]), "same bout", np.mean(bid[m] == bid[succ[m]]), "same sbj", np.mean(sbj[m] == sbj[succ[m]]))
print("score quantiles", np.quantile(d["score"][m], [0, .1, .5, .9, 1]))
