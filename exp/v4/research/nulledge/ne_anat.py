"""Error anatomy of the refined baseline: null<->activity errors, how many sit at chain label changes."""
import numpy as np
from ne_common import load_all, walks
from v4_local import macro_f1, N_CLS

d = load_all()
y = d["oof_y"]; lab = d["ref_o"]; fold = d["oof_fold"]; Qo = d["Qo"]; Po = d["Po"]
err = lab != y
print("errors", err.sum(), "of", len(y))
print("pred act / true null", ((lab > 0) & (y == 0)).sum(), " pred null / true act", ((lab == 0) & (y > 0)).sum(),
      " act/act", ((lab > 0) & (y > 0) & err).sum())
H = 3
Lo = d["Lo"]
near = np.zeros(len(y), bool); chg = np.zeros(len(y), bool)
for m in range(len(Lo)):
    F, B = walks(Lo[m], H + 1)
    c1 = ((F[0] >= 0) & (lab[np.maximum(F[0], 0)] != lab)) | ((B[0] >= 0) & (lab[np.maximum(B[0], 0)] != lab))
    chg |= c1
    nm = c1.copy()
    for k in range(H):
        for A in (F, B):
            g = A[k]; ok = g >= 0
            nm[ok] |= c1[g[ok]]
    near |= nm
print("change tiles", chg.sum(), "near (<=3)", near.sum())
for nmn, msk in (("change", chg), ("near", near)):
    print(nmn, "errors covered", (err & msk).sum(), "/", err.sum(), " null-act covered", (err & msk & ((lab == 0) | (y == 0))).sum(),
          "/", (err & ((lab == 0) | (y == 0))).sum(), "  null rate among cand", (y[msk] == 0).mean(), " pred null rate", (lab[msk] == 0).mean())
# for pred-null true-act: does Q non-null argmax match truth?
m = (lab == 0) & (y > 0)
qa = Qo[:, 1:].argmax(1) + 1
print("pred null / true act: Q nonnull argmax right", (qa[m] == y[m]).mean())
# neighbour label vote
votes = np.zeros((len(y), N_CLS))
for mm in range(len(Lo)):
    F, B = walks(Lo[mm], 3)
    for k in range(3):
        for A in (F, B):
            g = A[k]; ok = g >= 0
            np.add.at(votes, (np.flatnonzero(ok), lab[g[ok]]), 1.0 / (k + 1))
votes[:, 0] = 0
nv = votes.argmax(1)
print("pred null / true act: neighbour act vote right", (nv[m] == y[m]).mean(), " (has votes", (votes[m].sum(1) > 0).mean(), ")")
comb = np.log(Qo[:, 1:] + 1e-9) + 0.3 * np.log1p(votes[:, 1:])
print("combo right", ((comb.argmax(1) + 1)[m] == y[m]).mean())
# confusion pairs
from collections import Counter
c = Counter(zip(lab[err].tolist(), y[err].tolist()))
print(c.most_common(25))
