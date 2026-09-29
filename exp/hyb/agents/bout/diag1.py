"""diagnostics: are neighbour signals informative exactly where the classifier errs?"""
import numpy as np
from common import *

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
t = st // 50
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
bid = true_bouts(y, rec, st)
err = pred != y
# bout sizes, majority
u, inv, cnt = np.unique(bid, return_inverse=True, return_counts=True)
bmaj = np.zeros(len(u), np.int64); bl = np.zeros(len(u), np.int64)
for b in range(len(u)):
    ii = np.flatnonzero(inv == b); bmaj[b] = np.bincount(pred[ii], minlength=N_CLS).argmax(); bl[b] = y[ii[0]]
cat = np.where(~err, 0, np.where(bmaj[inv] != y, 3, 1))
print("errors", err.sum(), "whole-bout", np.mean(cat[err] == 3), "in majority-correct bouts", np.mean(cat[err] == 1))
# link time gaps
m = succ >= 0; dt = t[succ[m]] - t[m]; sr = rec[succ[m]] == rec[m]
print("link dt quantiles (same rec)", np.quantile(dt[sr], [.05, .1, .25, .5, .75, .9, .95]), "exact +1", np.mean(dt == 1), "|dt|<=3", np.mean(np.abs(dt) <= 3), "|dt|<=10", np.mean(np.abs(dt) <= 10), "same rec", sr.mean())
for q in (0.25, 0.5, 0.75):
    thr = np.quantile(score[m], q); mm = score[m] >= thr
    print(f" score>=q{q}: +1 {np.mean(dt[mm] == 1):.3f} |dt|<=3 {np.mean(np.abs(dt[mm]) <= 3):.3f} same bout {np.mean(bid[succ[m]][mm] == bid[m][mm]):.3f} same y {np.mean(y[succ[m]][mm] == y[m][mm]):.3f}")
for nm in ("raw_p0.0", "raw_p0.5"):
    nb = np.load(f"nb_{nm}.npy")
    for k in (1, 5, 10):
        nk = nb[:, :k]; dtt = np.abs(t[nk] - t[:, None]); sr2 = rec[nk] == rec[:, None]
        print(f"{nm} k={k}: |dt|<=3 {np.mean((dtt <= 3) & sr2):.3f} |dt|<=10 {np.mean((dtt <= 10) & sr2):.3f} same bout {np.mean(bid[nk] == bid[:, None]):.3f}")
    nk = nb[:, :10]
    for c, name in ((1, "err-in-good-bout"), (3, "whole-bout-err")):
        e = cat == c
        print(f"  {nm} {name}: nbr same bout {np.mean(bid[nk[e]] == bid[e][:, None]):.3f}, nbr pred==true {np.mean(pred[nk[e]] == y[e][:, None]):.3f}, nbr pred==own wrong {np.mean(pred[nk[e]] == pred[e][:, None]):.3f}, nbr true==own wrong pred {np.mean(y[nk[e]] == pred[e][:, None]):.3f}")
    e = cat == 0
    print(f"  {nm} correct: nbr same bout {np.mean(bid[nk[e]] == bid[e][:, None]):.3f}, nbr pred==true {np.mean(pred[nk[e]] == y[e][:, None]):.3f}")
# how contiguous are in-bout errors? run lengths of errors in true time order
runs = []
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); ii = ii[np.argsort(st[ii])]; e = err[ii] & (cat[ii] == 1)
    x = np.r_[0, e.astype(int), 0]; dd = np.diff(x); s0 = np.flatnonzero(dd == 1); s1 = np.flatnonzero(dd == -1); runs += list(s1 - s0)
runs = np.array(runs)
print("in-good-bout error runs: n", len(runs), "len quantiles", np.quantile(runs, [.5, .75, .9, .99]), "share of errs in runs>=5", runs[runs >= 5].sum() / runs.sum())
# per class of whole bout wrong: what labels
wb = [(bl[b], bmaj[b], cnt[b]) for b in range(len(u)) if bl[b] != bmaj[b] and cnt[b] >= 10]
print("whole-bout wrong bouts (>=10s):", len(wb), "true->pred", sorted(wb)[:60])
