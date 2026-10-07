"""quick look at the K7 arrays: link accuracy, recordings/sessions, count distribution"""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
from v4_local import load_fit, macro_f1, N_CLS, TRAIN_SETS
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
F = load_fit(K7)
y, sbj, fold, rec, start, ts = (F[k].astype(np.int64) for k in ("oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "true_succ"))
print("tiles", len(y), "subjects", np.unique(sbj).size, "folds", np.bincount(fold))
print("true_succ linked", (ts >= 0).mean(), "same-rec", np.mean(rec[ts[ts >= 0]] == rec[ts >= 0]))
for k in range(8):
    s = F["oof_succ"][k]; m = s >= 0
    print(f"member {k}: linked {m.mean():.3f} exact {(s[m] == ts[m]).mean():.4f} same-label {(y[s[m]] == y[m]).mean():.4f} score q10/50/90 {np.percentile(F['oof_score'][k][m], [10, 50, 90]).round(2)}")
s0 = F["oof_succ"][0]; m = s0 >= 0; ok = s0 == ts
print("score of correct vs wrong links (member0):", np.median(F["oof_score"][0][m & ok]), np.median(F["oof_score"][0][m & ~ok]))
print("test links linked", (F["test_succ"][0] >= 0).mean())
for s in np.unique(sbj):
    ii = sbj == s; rr = np.unique(rec[ii])
    desc = []
    for r in rr:
        jj = ii & (rec == r); cl = np.unique(y[jj]); cl = cl[cl > 0]
        desc.append(f"rec{r}:{jj.sum()}t/{len(cl)}c")
    print(f"sbj {s:2d} fold {fold[ii][0]} n={ii.sum()} " + " ".join(desc))
true = np.array([[(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for c in range(N_CLS)] for s in np.unique(sbj)])
print("null frac per subject", np.round(true[:, 0] / true.sum(1), 3))
print("exercise counts: mean", true[:, 1:].mean().round(1), "std", true[:, 1:].std().round(1), "min", true[:, 1:].min(), "max", true[:, 1:].max())
print("per-class mean count", np.round(true[:, 1:].mean(0), 0))
print("B2 tile F1", macro_f1(y, F["B2_OOF"].argmax(1)), "QB", macro_f1(y, F["QB_OOF"].argmax(1)), "ref", macro_f1(y, F["ref_oof"]))
# class sets per recording: are the 9-activity blocks fixed across subjects?
from collections import Counter
blocks = Counter()
for s in np.unique(sbj):
    for r in np.unique(rec[sbj == s]):
        jj = (sbj == s) & (rec == r); cl = tuple(sorted(set(np.unique(y[jj]).tolist()) - {0}))
        blocks[cl] += 1
for b, n in blocks.most_common(12):
    print(n, b)
