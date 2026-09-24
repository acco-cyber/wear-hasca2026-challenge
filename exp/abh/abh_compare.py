"""Compare abh test probs vs the notebook's public submission.csv and our current best submission.
python abh_compare.py <probs.npy> [more.npy ...]"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

NB = r"E:\Claude code\wear\public_subs\abhinavm2811__3rd-wear-dataset-challenge-hasca-2026\submission.csv"
BEST = r"E:\Claude code\wear\subs\sub_transductive_mrf4_e19_aka045_abh.csv"
CAL_W = np.array([0.75, 0.85, 1., 0.8, 0.55, 0.75, 1.55, 1.5, 1.25, 0.5, 1.2, 2., 1.45, 1.25, 0.95, 0.75, 0.8, 0.8, 1.95])


def load(p):
    d = pd.read_csv(p).sort_values("id"); assert (d.id.to_numpy() == np.arange(12234)).all()
    return d.target_feature.to_numpy().astype(int)


nb = load(NB); best = load(BEST)
print("notebook sub null share %.3f  dist %s" % ((nb == 0).mean(), np.bincount(nb, minlength=19).tolist()))
print("notebook sub vs best: agree %.4f  macroF1 %.4f" % ((nb == best).mean(), f1_score(best, nb, average="macro")))
for p in sys.argv[1:]:
    P = np.load(p); assert P.shape == (12234, 19), P.shape
    for name, a in (("raw", P.argmax(1)), ("cal", (P * CAL_W).argmax(1))):
        print(f"{p} [{name}] agree_nb {(a == nb).mean():.4f}  agree_best {(a == best).mean():.4f}  "
              f"F1_vs_best {f1_score(best, a, average='macro'):.4f}  F1_vs_nb {f1_score(nb, a, average='macro'):.4f}  "
              f"null {(a == 0).mean():.3f}")
    print("   dist", np.bincount(P.argmax(1), minlength=19).tolist(), " mean maxprob %.3f" % P.max(1).mean())
