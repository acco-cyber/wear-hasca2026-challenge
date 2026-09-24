import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.metrics import f1_score, confusion_matrix
p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EXP, "blend_v1f", "oof.npy")
oof = np.load(p); eval_single(oof, name=p)
import common
m = meta(); y = m.y.to_numpy(); rl = common._rl; ok = (rl >= 0) & (m.pur.to_numpy() >= 0.8); idx = np.where(ok)[0]
P = oof[idx, rl[idx]]; pr = P.argmax(1); yt = y[idx]
print("per-class F1:", np.round(f1_score(yt, pr, average=None), 3).tolist())
print("family F1:", np.round(f1_score(FAMILY[yt], FAMILY[pr], average=None), 3).tolist(), "macro", round(f1_score(FAMILY[yt], FAMILY[pr], average="macro"), 4))
# within-family accuracy given family correct
fc = FAMILY[yt] == FAMILY[pr]
print("variant acc | family correct:", round(float((yt[fc] == pr[fc]).mean()), 4), "family acc", round(float(fc.mean()), 4))
np.set_printoptions(linewidth=250)
C = confusion_matrix(yt, pr); print((C / C.sum(1, keepdims=True) * 100).round(0).astype(int))
# per-limb F1 (all limbs)
for li in range(4):
    v = ~np.isnan(oof[:, li, 0]); print("limb", LIMBS[li], round(f1_score(y[v], oof[v, li].argmax(1), average="macro"), 4))
