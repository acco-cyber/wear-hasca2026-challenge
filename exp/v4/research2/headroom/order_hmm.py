"""Ceiling of PERFECT time order (oracle) with a plain sequence decoder instead of the graph: per recording, tiles in
true order, Viterbi with a sticky transition matrix (stay 1-eps) on log emissions. eps chosen NESTED by subject fold
(best on the other 4 folds). Emissions: stage-B base B2 (pre-graph), K7 graph P (pass 2), fused b4wa Q."""
import numpy as np
from hlib import *

D = setup(); y, fold = D["y"], D["fold"]
z = np.load(os.path.join(HERE, "k7_base.npz"))
EM = {"B2 base": D["Bo"].astype(np.float64), "K7 graph P": np.log(np.clip(z["P"].astype(np.float64), 1e-9, None)),
      "b4wa Q": np.log(np.clip(np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64), 1e-9, None))}


def viterbi(E, eps):
    T, K = E.shape; ls, lw = np.log(1 - eps), np.log(eps / (K - 1))
    dp = E[0].copy(); bp = np.zeros((T, K), np.int32)
    for t in range(1, T):
        best = dp.max(); arg = dp.argmax()
        stay = dp + ls; sw = best + lw
        choose_sw = sw > stay; bp[t] = np.where(choose_sw, arg, np.arange(K))
        dp = np.where(choose_sw, sw, stay) + E[t]
    out = np.zeros(T, np.int64); out[-1] = dp.argmax()
    for t in range(T - 1, 0, -1):
        out[t - 1] = bp[t, out[t]]
    return out


EPS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-6]
for nm, E in EM.items():
    labs = {}
    for eps in EPS:
        lab = np.zeros(len(y), np.int64)
        for ii in D["order"]:
            lab[ii] = viterbi(E[ii], eps)
        labs[eps] = lab
    per = {eps: [macro_f1(y[fold == f], labs[eps][fold == f]) for f in range(FOLDS)] for eps in EPS}
    # nested choice: for each fold, the eps with the best mean F1 on the other folds
    nest = np.zeros(len(y), np.int64); picks = []
    for f in range(FOLDS):
        e_best = max(EPS, key=lambda e: np.mean([per[e][g] for g in range(FOLDS) if g != f])); picks.append(e_best)
        nest[fold == f] = labs[e_best][fold == f]
    log(f"{nm}: argmax {macro_f1(y, E.argmax(1)):.4f} | " + " ".join(f"eps{e:g}:{macro_f1(y, labs[e]):.4f}" for e in EPS)
        + f" | NESTED {macro_f1(y, nest):.4f} (picks {picks})")
