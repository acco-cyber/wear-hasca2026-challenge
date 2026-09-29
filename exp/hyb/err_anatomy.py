"""Error anatomy of the Hanbat-graph OOF output on their 69,326 train tiles (true time order known via rec/start):
- true bouts (runs of one label in true order), whole-bout errors vs boundary vs scattered errors
- do subjects do each exercise exactly once? duration distribution? is the activity ORDER shared across sessions?
- oracle: if every true bout got its majority predicted label / its correct label, what macro-F1?
python err_anatomy.py <their_oof_P.npy>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, KEEP, TRAIN_SETS, macro_f1, N_CLS
P = np.load(sys.argv[1]).astype(np.float64); P /= P.sum(1, keepdims=True)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
y, rec, st, sbj = sm["y"], sm["rec"], sm["start"], sm["sbj"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
print("OOF macro F1", round(macro_f1(y, pred), 4), "acc", round(float(np.mean(pred == y)), 4))
cat = np.zeros(len(y), int)   # 0 ok, 1 boundary(<=5s from bout edge), 2 inside bout w/ correct majority, 3 whole-bout wrong majority
orders = {}; bout_stats = []; maj_lab = pred.copy()
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); ii = ii[np.argsort(st[ii])]
    yy = y[ii]; pp = pred[ii]
    # runs
    cut = np.flatnonzero(np.diff(yy) != 0) + 1; starts = np.r_[0, cut]; ends = np.r_[cut, len(yy)]
    seq = []
    for a, b in zip(starts, ends):
        lab = yy[a]; L = b - a; seg = pp[a:b]
        maj = np.bincount(seg, minlength=N_CLS).argmax()
        if lab != 0 and L >= 10:
            seq.append(lab)
        pos = np.arange(L); edge = np.minimum(pos, L - 1 - pos)
        wrong = seg != lab
        for k in np.flatnonzero(wrong):
            cat[ii[a + k]] = 3 if maj != lab else (1 if edge[k] <= 5 else 2)
        bout_stats.append((r, lab, L, maj == lab, np.mean(seg == lab)))
        maj_lab[ii[a:b]] = maj
    orders[r] = seq
bs = np.array([(b[1], b[2], b[3], b[4]) for b in bout_stats], dtype=float)
act = bs[(bs[:, 0] > 0) & (bs[:, 1] >= 10)]
print(f"\nactivity bouts (>=10s): {len(act)}; median length {np.median(act[:, 1]):.0f}s; majority-correct {act[:, 2].mean():.3f}; mean within-bout acc {act[:, 3].mean():.3f}")
for c in range(1, N_CLS):
    m = act[:, 0] == c
    print(f"  class {c:2d}: bouts {int(m.sum()):2d}  len median {np.median(act[m, 1]) if m.any() else 0:5.0f}  maj-correct {act[m, 2].mean() if m.any() else 0:.2f}  acc {act[m, 3].mean() if m.any() else 0:.2f}")
err = pred != y
print(f"\nerrors {err.sum()} ({err.mean():.3f}): boundary(<=5s) {np.mean(cat[err] == 1):.3f}, inside-bout {np.mean(cat[err] == 2):.3f}, whole-bout-wrong {np.mean(cat[err] == 3):.3f}")
print("error rows by true class null vs activity:", f"null {np.mean(y[err] == 0):.3f}")
# bouts per class per session (exactly once?)
print("\nactivity bouts per class per session (>=10s):")
cnt = {}
for r, seq in orders.items():
    u, c = np.unique(seq, return_counts=True); cnt[r] = dict(zip(u.tolist(), c.tolist()))
multi = sum(v > 1 for d in cnt.values() for v in d.values()); total = sum(len(d) for d in cnt.values())
print(f"  (class,session) pairs {total}, with >1 bout {multi}; sessions missing classes: " + str({int(r): 18 - len(d) for r, d in cnt.items() if len(d) < 18}))
# order agreement across sessions: first-occurrence order of the 18 classes
firsts = {}
for r, seq in orders.items():
    o = []
    for c in seq:
        if c not in o: o.append(c)
    firsts[r] = o
    print(f"  rec {r:2d} order: {o}")
# pairwise Kendall-like agreement of class order
def rank(o):
    return {c: i for i, c in enumerate(o)}
rs = list(firsts); agree = []
for i in range(len(rs)):
    for j in range(i + 1, len(rs)):
        a, b = rank(firsts[rs[i]]), rank(firsts[rs[j]]); cs = [c for c in a if c in b]
        conc = sum((a[x] < a[z]) == (b[x] < b[z]) for k, x in enumerate(cs) for z in cs[k + 1:]); n = len(cs) * (len(cs) - 1) / 2
        agree.append(conc / max(n, 1))
print(f"\nmean pairwise order concordance across sessions: {np.mean(agree):.3f} (0.5 = random, 1 = identical order)")
print("oracle: each true bout -> its majority predicted label: macro F1", round(macro_f1(y, maj_lab), 4))
