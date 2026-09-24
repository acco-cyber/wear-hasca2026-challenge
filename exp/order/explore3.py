"""Error analysis of e7 labels: per true activity -> distribution of e7 labels; oracle within-family fix upper bound."""
import os, sys, pickle
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
FAM = np.zeros(19, int); FAM[1:6] = 1; FAM[6:11] = 2; FAM[11:13] = 3; FAM[13:15] = 4; FAM[15] = 5; FAM[16:18] = 6; FAM[18] = 7
SESS = {"eval": ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"],
        "extra": ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"],
        "extra2": ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]}
allS = sum(SESS.values(), [])
C = np.zeros((19, 19), int)
res = {}
for s in allS:
    y = R[s]["y"]; lab = R[s]["lab"]; C += confusion_matrix(y, lab, labels=range(19))
    # oracle A: within-family fix (if fam(lab)==fam(y) and y>0 -> y)
    la = lab.copy(); m = (FAM[lab] == FAM[y]) & (y > 0); la[m] = y[m]
    # oracle B: null<->activity fix only
    lb = lab.copy(); m = (lab == 0) | (y == 0); lb[m] = y[m]
    # oracle C: family-level errors fixed (cross-family) but keep within-family error
    res[s] = (f1_score(y, lab, average="macro"), f1_score(y, la, average="macro"), f1_score(y, lb, average="macro"))
for w, ss in SESS.items():
    a = np.array([res[s] for s in ss]).mean(0); print(w, "e7 %.4f  fix-within-family %.4f  fix-null %.4f" % tuple(a))
a = np.array([res[s] for s in allS]).mean(0); print("all18", "e7 %.4f  fix-within-family %.4f  fix-null %.4f" % tuple(a))
np.set_printoptions(linewidth=250)
print("confusion (rows true, cols e7), 18 sessions summed")
print("     " + " ".join(f"{j:5d}" for j in range(19)))
for i in range(19): print(f"{i:3d}: " + " ".join(f"{C[i,j]:5d}" for j in range(19)))
# per-session within-family swap magnitude per family
for s in allS:
    y = R[s]["y"]; lab = R[s]["lab"]; out = []
    for f, cls in [(1, range(1, 6)), (2, range(6, 11)), (3, (11, 12)), (4, (13, 14)), (6, (16, 17))]:
        m = (FAM[y] == f) & (FAM[lab] == f); wrong = np.sum(m & (lab != y))
        out.append(f"f{f}:{wrong}/{m.sum()}")
    print(s, " ".join(out), "| null->act", int(np.sum((y == 0) & (lab > 0))), "act->null", int(np.sum((y > 0) & (lab == 0))))
