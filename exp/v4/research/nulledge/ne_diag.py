"""Diagnostics of the outer-fold specialist probabilities: per subject / per flip type, what fixed thresholds would do
(descriptive only, not used for the reported score)."""
import os, sys
import numpy as np
from ne_common import HERE, K7
from v4_local import macro_f1, FOLDS

tag = sys.argv[1] if len(sys.argv) > 1 else "null_cand3"
mode = tag.split("_")[0]
r = np.load(os.path.join(HERE, f"res_{tag}.npz")); z = np.load(os.path.join(HERE, "feats.npz")); b = np.load(os.path.join(HERE, "base_cache.npz"))
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
y = st["oof_y"]; fold = st["oof_fold"]; sbj = st["oof_sbj"]; lab = b["ref_o"]; qa = z["qao"]
idx, p = r["idx"], r["p_outer"]
cur = lab[idx]; yt = y[idx]
from sklearn.metrics import roc_auc_score
if mode == "null":
    a = cur > 0
    print("act tiles: AUC p(null) vs y==0", roc_auc_score(yt[a] == 0, p[a]), " base null rate", (yt[a] == 0).mean())
    n = cur == 0
    print("null tiles: AUC 1-p vs y==qa", roc_auc_score(yt[n] == qa[idx][n], 1 - p[n]), " rate y==qa", (yt[n] == qa[idx][n]).mean())
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = a & (p > t); print(f"act->null p>{t}: {m.sum()} flips, right {(yt[m] == 0).sum()}, was right {(yt[m] == cur[m]).sum()}")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        m = n & (p < t); print(f"null->act p<{t}: {m.sum()} flips, right {(yt[m] == qa[idx][m]).sum()}, was right {(yt[m] == 0).sum()}")
for k in range(FOLDS):
    for s in np.unique(sbj[fold == k]):
        ii = sbj[idx] == s
        if mode == "null":
            m1 = ii & (cur > 0) & (p > 0.6); m2 = ii & (cur == 0) & (p < 0.4)
            print(f"fold {k} sbj {s:2d}: act->null p>.6 {m1.sum():4d} (right {(yt[m1] == 0).sum():4d}, broke {(yt[m1] == cur[m1]).sum():4d}) | "
                  f"null->act p<.4 {m2.sum():4d} (right {(yt[m2] == qa[idx][m2]).sum():4d}, broke {(yt[m2] == 0).sum():4d})")
