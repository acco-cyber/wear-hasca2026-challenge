"""How predictable is the next bout? Nested (leave-fold-out) top-1 / top-3 accuracy and mean log P of the empirical
bout-transition model on the held-out fold's true bout sequences; also the same along the decoded chains."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_tr import bout_sequences, trans_logp, chain_context
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "cache.npz"))
y, fold, rec, start = z["y"], z["fold"], z["oof_rec"], z["oof_start"]
for alpha in (0.1, 0.5, 2.0):
    t1 = t3 = n = 0; ll = 0.0; llu = 0.0
    for k in range(5):
        lf, lb = trans_logp(bout_sequences(y, rec, start, fold != k), alpha)
        for s in bout_sequences(y, rec, start, fold == k):
            for a, b in zip(s[:-1], s[1:]):
                order = np.argsort(-lf[a]); order = order[lf[a, order] != 0]
                t1 += order[0] == b; t3 += b in order[:3]; n += 1; ll += lf[a, b]; llu += np.log(1 / 17)
    print(f"alpha {alpha}: {n} held-out transitions, top1 {t1 / n:.3f}, top3 {t3 / n:.3f}, mean logP {ll / n:.3f} (uniform {llu / n:.3f})")
# decoded chain: how often is the decoded (prev bout -> bout) pair the true pair
fin = z["fin_o"]; succ = z["oof_succ"][0]
pb, nb, rl, bl = chain_context(fin, succ)
pbt, nbt, _, _ = chain_context(y, z["true_succ"])
m = (fin > 0) & (fin == y)
print("correct non-null tiles: decoded prev bout == true prev bout", np.mean(pb[m] == pbt[m]), "next", np.mean(nb[m] == nbt[m]))
pbo, nbo, _, _ = chain_context(y, succ)
print("true labels along matching 0: prev bout == true", np.mean(pbo[y > 0] == pbt[y > 0]))
