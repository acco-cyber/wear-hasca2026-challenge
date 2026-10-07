"""Where do the stage-C changes land? per subject / per class deltas of a prediction file vs the refined b4wa labels."""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1
HERE = os.path.dirname(os.path.abspath(__file__))
tag, key = sys.argv[1], sys.argv[2]
z = np.load(os.path.join(HERE, "cache", "feat.npz")); p = np.load(os.path.join(HERE, "cache", f"pred_{tag}.npz"))
y, sbj, fold, lo = z["y"], z["sbj"], z["fold"], z["lo"]; new = p[key + "_o"]
ch = new != lo
print(f"{tag}/{key}: changed {ch.sum()} tiles; fixed {np.sum(ch & (new == y))}, broke {np.sum(ch & (lo == y))}, wrong->wrong {np.sum(ch & (new != y) & (lo != y))}")
print("F1", round(macro_f1(y, lo), 4), "->", round(macro_f1(y, new), 4))
for s in np.unique(sbj):
    ii = sbj == s
    c = ch[ii].sum()
    print(f"  sbj {s:2d} fold {fold[ii][0]} n {ii.sum():5d} changed {c:4d} fixed {np.sum(ch[ii] & (new[ii] == y[ii])):4d} broke {np.sum(ch[ii] & (lo[ii] == y[ii])):4d} "
          f"acc {np.mean(lo[ii] == y[ii]):.4f}->{np.mean(new[ii] == y[ii]):.4f}")
fr = np.bincount(lo[ch], minlength=19); to = np.bincount(new[ch], minlength=19)
print("changes from-class", fr.tolist()); print("changes to-class  ", to.tolist())
# per-class F1
def pcf(pred):
    out = []
    for c in range(19):
        tp = np.sum((pred == c) & (y == c)); out.append(2 * tp / max(np.sum(pred == c) + np.sum(y == c), 1))
    return np.array(out)
print("per-class F1 delta", np.round(pcf(new) - pcf(lo), 4).tolist())
