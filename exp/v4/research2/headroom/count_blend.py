"""A label-free count tweak, finish-only on K7's 2-pass P: exercise targets pulled towards the graph's own evidence
(argmax count / soft count of P before Sinkhorn): t = cnt + a * (ev - cnt), null = rest. a chosen NESTED by subject fold."""
import numpy as np
from hlib import *

D = setup(); y, sbj, fold = D["y"], D["sbj"], D["fold"]
P = np.load(os.path.join(HERE, "k7_base.npz"))["P"].astype(np.float64)
cnt, key, true = learned_counts(D, P)
ns = np.array([TRAIN_SETS.get(int(s), 1) for s, _ in key])
am = np.array([(P[sbj == s].argmax(1) == c).sum() for s, c in key]) / ns
Q2 = np.clip(P, 1e-12, None) ** 2; Q2 /= Q2.sum(1, keepdims=True)
soft = np.array([Q2[sbj == s, c].sum() for s, c in key]) / ns
log(f"MAE: regressor {np.abs(cnt - true).mean():.2f}, P-argmax count {np.abs(am - true).mean():.2f}, sharpened soft count {np.abs(soft - true).mean():.2f}")
A = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
for nm, ev in (("argmax", am), ("soft", soft)):
    labs = {}
    for a in A:
        c = cnt + a * (ev - cnt); labs[a] = finish_targets(P, sbj, targets_from(D, key, c)).argmax(1)
    per = {a: [macro_f1(y[fold == f], labs[a][fold == f]) for f in range(FOLDS)] for a in A}
    nest = np.zeros(len(y), np.int64); picks = []
    for f in range(FOLDS):
        a_b = max(A, key=lambda a: np.mean([per[a][g] for g in range(FOLDS) if g != f])); picks.append(a_b); nest[fold == f] = labs[a_b][fold == f]
    log(f"{nm}: " + " ".join(f"a{a}:{macro_f1(y, labs[a]):.4f}" for a in A) + f" | NESTED {macro_f1(y, nest):.4f} picks {picks}")
