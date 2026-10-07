"""How much of the remaining OOF error is 'class-level' (a subject's whole class predicted as another class), i.e. the
kind of error that one Kaggle submission can test as a per-subject label permutation? Read-only on the best fused
decode (sub_v4c_b4wa_labo.npy, OOF 0.9336). Also: macro-F1 if every such class-level confusion were fixed (oracle),
and how the model ranks the true swaps (log-likelihood ratio from Qo)."""
import os
import numpy as np
from sklearn.metrics import f1_score

W = r"E:\Claude code\wear"
st = np.load(os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4", "stage.npz"))
y, sbj = st["oof_y"], st["oof_sbj"]
lab = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labo.npy"))
Q = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_Qo.npy"))
print("shapes", lab.shape, Q.shape, "OOF macro-F1", round(f1_score(y, lab, average="macro"), 4))
fixed = lab.copy(); rows = []
for s in np.unique(sbj):
    m = sbj == s
    for c in range(1, 19):
        tc = m & (y == c)
        if tc.sum() == 0:
            continue
        pred = np.bincount(lab[tc], minlength=19)
        top = pred.argmax(); share = pred[top] / tc.sum()
        if top != c and top != 0:
            rows.append((s, c, top, tc.sum(), round(share, 2)))
            # oracle fix: those tiles predicted 'top' inside the true class get c, and tiles truly 'top' predicted c get top
            fixed[tc & (lab == top)] = c
            tb = m & (y == top)
            fixed[tb & (lab == c)] = top
print("class-level confusions (subject, true, predicted, n, share):")
for r in rows:
    print("  ", r)
print("n subjects", len(np.unique(sbj)), "| class-level confusions", len(rows), f"-> {len(rows) / len(np.unique(sbj)):.2f} per subject")
print("OOF macro-F1 after oracle fix of those:", round(f1_score(y, fixed, average="macro"), 4))
# per-subject macro-F1 deltas
d = []
for s in np.unique(sbj):
    m = sbj == s
    d.append(f1_score(y[m], fixed[m], average="macro") - f1_score(y[m], lab[m], average="macro"))
print("per-subject macro-F1 gain from the oracle fix: mean", round(np.mean(d), 4), "max", round(np.max(d), 4))
# null-vs-activity share of remaining errors
err = lab != y
print("error share: null->act", round(np.mean((y[err] == 0) & (lab[err] != 0)), 3), "act->null", round(np.mean((y[err] != 0) & (lab[err] == 0)), 3), "act->act", round(np.mean((y[err] != 0) & (lab[err] != 0)), 3))
