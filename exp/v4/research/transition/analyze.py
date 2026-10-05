"""Headroom analysis: error types of the fused pre-refiner labels along the TRUE tile order."""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
from v4_local import macro_f1
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "cache.npz"))
y, fin, rec, st, fold = z["y"], z["fin_o"], z["oof_rec"], z["oof_start"], z["fold"]
succ = z["oof_succ"]; ts = z["true_succ"]
for k in range(len(succ)):
    m = succ[k] >= 0
    print(k, "link exact", round(float((succ[k][m] == ts[m]).mean()), 4), "coverage", round(float(m.mean()), 4))
# errors relative to true boundaries
err = fin != y
dist = np.full(len(y), 999)
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); o = ii[np.argsort(st[ii])]; lab = y[o]
    ch = np.flatnonzero(np.diff(lab) != 0)  # boundary between o[c] and o[c+1]
    pos = np.arange(len(o))
    d = np.full(len(o), 999)
    for c in ch:
        d = np.minimum(d, np.minimum(np.abs(pos - c), np.abs(pos - c - 1)))
    dist[o] = d
print("errors", err.sum(), "of", len(y))
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 5), (6, 20), (21, 999)]:
    m = (dist >= lo) & (dist <= hi)
    print(f"dist {lo}-{hi}: tiles {m.sum()}, errors {(err & m).sum()}, err rate {(err & m).sum() / max(m.sum(), 1):.3f}")
# error types: null<->act, act<->act
print("y null, pred act", (err & (y == 0)).sum(), "| y act, pred null", (err & (y > 0) & (fin == 0)).sum(), "| act<->act", (err & (y > 0) & (fin > 0)).sum())
# whole-bout errors: true non-null bouts, fraction mislabeled
bad = []
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); o = ii[np.argsort(st[ii])]; lab = y[o]; pr = fin[o]
    ch = np.r_[0, np.flatnonzero(np.diff(lab) != 0) + 1, len(lab)]
    for a, b in zip(ch[:-1], ch[1:]):
        if lab[a] == 0:
            continue
        acc = (pr[a:b] == lab[a]).mean()
        bad.append((b - a, acc, (pr[a:b] != lab[a]).sum()))
bad = np.array(bad)
for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
    m = (bad[:, 1] >= lo) & (bad[:, 1] < hi)
    print(f"true bouts with acc {lo}-{hi}: {m.sum()} bouts, {bad[m, 2].sum():.0f} wrong tiles, mean len {bad[m, 0].mean() if m.any() else 0:.1f}")
# confusion pairs
from collections import Counter
c = Counter(zip(y[err], fin[err]))
print(c.most_common(25))
