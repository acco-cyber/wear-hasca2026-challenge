import pickle
import numpy as np
from common import *

d = load_oof(); y, sbj, fold = d["y"], d["sbj"], d["fold"]
pred = finish(d["P"], dict(sbj=sbj, sets=TRAIN_SETS)); base = macro_f1(y, pred)
outs = pickle.load(open("icm2_outs.pkl", "rb"))
for src in ("q", "p", None):
    grid = [g for g in outs if src is None or g[4] == src]
    nest = pred.copy(); ch = {}
    for f in np.unique(fold):
        tr = fold != f; best = max(grid, key=lambda g: macro_f1(y[tr], outs[g][tr])); ch[int(f)] = best; nest[fold == f] = outs[best][fold == f]
    print(f"src={src}: NESTED F1 {macro_f1(y, nest):.4f} (+{macro_f1(y, nest) - base:.4f}) {per_fold(y, nest, fold)} chosen {ch}")
# distribution over q grid
fq = sorted(((macro_f1(y, o), g) for g, o in outs.items() if g[4] == "q"), reverse=True)
print("q-grid F1: min", round(fq[-1][0], 4), "median", round(np.median([f for f, _ in fq]), 4), "max", round(fq[0][0], 4), "n", len(fq))
print("top q:", [(round(f, 4), g) for f, g in fq[:6]])
g0 = (1.0, 5, 4.0, 0.0, "q"); o = outs[g0]
print("chosen fixed config", g0, round(macro_f1(y, o), 4), per_fold(y, o, fold))
# per-subject deltas
ds = [(int(s), round(macro_f1(y[sbj == s], o[sbj == s]) - macro_f1(y[sbj == s], pred[sbj == s]), 4)) for s in np.unique(sbj)]
print("per-subject delta", ds, "n improved", sum(v > 0 for _, v in ds), "worse", sum(v < 0 for _, v in ds))
from sklearn.metrics import f1_score
pc0 = f1_score(y, pred, average=None, labels=range(N_CLS)); pc1 = f1_score(y, o, average=None, labels=range(N_CLS))
print("per-class delta", np.round(pc1 - pc0, 4))
print("fixed:", np.sum((o == y) & (pred != y)), "broken:", np.sum((o != y) & (pred == y)))
