"""Error structure of the baseline decoder on eval sessions."""
import numpy as np
from common import *
from sklearn.metrics import confusion_matrix
oof = load_oof(0.2); S = load_structs(("eval",))
CM = np.zeros((NC, NC), int); CMr = np.zeros((NC, NC), int)
for s in EVAL:
    st = S[s]; P = sess_P(oof, st); y = st["y"]; lab = baseline(P, st)
    CM += confusion_matrix(y, lab, labels=np.arange(NC)); CMr += confusion_matrix(y, P.argmax(1), labels=np.arange(NC))
    cnt = np.bincount(lab, minlength=NC); ct = np.bincount(y, minlength=NC)
    print(s, "pred counts", cnt.tolist()); print(s, "true counts", ct.tolist())
np.set_printoptions(linewidth=250)
print("baseline confusion (rows true, cols pred)"); print(CM)
tot = CM.sum(); off = CM.copy(); np.fill_diagonal(off, 0)
print("errors total", off.sum(), "of", tot)
print("  true null -> act", off[0, 1:].sum(), "| true act -> null", off[1:, 0].sum(), "| act -> other act", off[1:, 1:].sum())
fam = np.array([0] + [1] * 5 + [2] * 5 + [3, 3, 4, 4, 5, 6, 6, 7])
within = sum(off[i, j] for i in range(1, NC) for j in range(1, NC) if fam[i] == fam[j]); print("  act->act within family", within)
