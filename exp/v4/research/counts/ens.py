"""Averages of the saved diagnostic count predictions + subject-mean shrinkage (diagnostic)."""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import FOLDS, macro_f1, N_CLS

D = C.load(); y, fold = D["y"], D["fold"]
X, key, Xt, kt, true, kf = C.build(D, [D["Po"], D["Bpo"]], [D["Pt"], D["Bpt"]])
P = {os.path.basename(p)[9:-4]: np.load(p) for p in glob.glob(os.path.join(C.HERE, "diag_cnt_*.npy"))}
def rep(nm, c):
    f, lab = C.f1_from_counts(D, D["Po"], key, c)
    print(f"{nm:40s} err {np.abs(c - true).mean():5.2f} cen {C.centred_err(c, true, key):5.2f} F1 {f:.4f} | "
          + " ".join(f"{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(FOLDS)), flush=True)
sets = {
    "all non-wide": [k for k in P if not k.endswith("_w") and "cen" not in k],
    "F4/E4/F7/E7 non-wide": [k for k in P if k[:2] in ("F4", "E4", "F7", "E7") and not k.endswith("_w") and "cen" not in k],
    "F0 ridge+ridge_log+lgb": ["F0_ridge", "F0_ridge_log", "F0_lgb"],
    "F4 ridge_log+lgb": ["F4_ridge_log", "F4_lgb"],
    "F0rl+F4rl+F4lgb+F7rl+E4rl": ["F0_ridge_log", "F4_ridge_log", "F4_lgb", "F7_ridge_log", "E4_ridge_log"],
}
for nm, ks in sets.items():
    ks = [k for k in ks if k in P]; rep(f"{nm} ({len(ks)})", np.mean([P[k] for k in ks], 0))
base = P["F0_ridge"]
for lam in (0.8, 0.9, 1.1, 1.2):            # scale the within-subject deviation (diag)
    c = base.copy()
    for s in np.unique(key[:, 0]):
        m = key[:, 0] == s; c[m] = base[m].mean() + lam * (base[m] - base[m].mean())
    rep(f"F0 ridge, within-subject dev x{lam}", c)
