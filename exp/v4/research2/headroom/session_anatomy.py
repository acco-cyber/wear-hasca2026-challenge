"""What does the 2-half session oracle fix on the fused labels? breakdown of the violating tiles, block accuracy of the
decoded activity labels, and a LABEL-FREE proxy: the session of a tile estimated from the decoded labels of its links
and video kNN neighbours (no true order, no labels) -> how often does it agree with the oracle session's block?"""
import numpy as np
from collections import Counter
from hlib import *

B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; BLK = np.zeros(19, int); BLK[B1] = 1; BLK[[c for c in range(1, 19) if c not in B1]] = 2
D = setup(); y, sbj = D["y"], D["sbj"]; n = len(y)
M = np.load(os.path.join(HERE, "masks.npz")); MS = M["MS"].astype(np.float64); ses = M["ses"]
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64); lab = np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64)
bad = MS[np.arange(n), lab] == 0
Qm = Qo * MS; la = Qm.argmax(1)
log(f"violating tiles {bad.sum()}: true null {np.sum(bad & (y == 0))}, true activity {np.sum(bad & (y > 0))}; "
    f"masked argmax correct on {np.sum(bad & (la == y))}")
log("violating (true, decoded) top: " + str(Counter(zip(y[bad].tolist(), lab[bad].tolist())).most_common(12)))
log("violating tiles per subject: " + str(sorted(Counter(sbj[bad].tolist()).items())))
# block of the oracle session (majority block of its true activity tiles)
sblk = {s_: np.bincount(BLK[y[(ses == s_) & (y > 0)]], minlength=3).argmax() for s_ in np.unique(ses)}
tb = np.array([sblk[s_] for s_ in ses])
act = lab > 0
log(f"decoded activity tiles whose class block == oracle session block: {np.mean(BLK[lab[act]] == tb[act]):.4f} ({np.sum(BLK[lab[act]] != tb[act])} tiles)")
# label-free proxy: video kNN (whitened subject embedding, within subject) majority of decoded blocks among 20 neighbours
E = D["Eo"]; prox = np.zeros(n, int)
for s in np.unique(sbj):
    ii = np.flatnonzero(sbj == s); S = E[ii] @ E[ii].T; np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, 20, axis=1)[:, :20]; bl = BLK[lab[ii][nb]]
    c1, c2 = (bl == 1).sum(1), (bl == 2).sum(1); prox[ii] = np.where(c1 > c2, 1, np.where(c2 > c1, 2, 0))
for nm, m in (("all tiles", np.ones(n, bool)), ("decoded-activity", act), ("violating", bad)):
    ok = prox[m] == tb[m]
    log(f"kNN-block proxy vs oracle session block on {nm}: agree {ok.mean():.4f} (undecided {np.mean(prox[m] == 0):.3f})")
