import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as R, exp as E, exp2 as E2

d = R.load_cache(); y, fin = d["y"], d["fin_o"]
so = E.Side(d, "oof"); Eu = E2.centred_unit(d["vmean_o"], d["sbj"])
X, g, oth = E2.rows_one2(so, fin, 0, 3, ("fit", "link", "vote"), True, Eu)
T = ((y[g] == oth) & (y[g] != fin[g])).astype(int)
print(X.shape, T.mean())


def auc(s, t):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = t.sum(); return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(t) - n1))


for j in range(X.shape[1]):
    print(j, f"auc {auc(X[:, j], T):.3f}", f"mean {X[:, j].mean():.3f}", f"min {X[:, j].min():.2f} max {X[:, j].max():.2f}")
