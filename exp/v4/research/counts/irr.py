"""Evidence available for the irregular keys (diagnostic) + label-free target iteration test."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import FOLDS, macro_f1, N_CLS, TRAIN_SETS

D = C.load(); y, fold, sbj, Po = D["y"], D["fold"], D["sbj"], D["Po"]
X, key, Xt, kt, true, kf = C.build(D, [D["Po"], D["Bpo"]], [D["Pt"], D["Bpt"]])
cnt = np.load(os.path.join(C.HERE, "diag_cnt_F0_ridge.npy"))
f0, lab0 = C.f1_from_counts(D, Po, key, cnt)
ns_ = np.array([TRAIN_SETS.get(int(s), 1) for s, _ in key])
def cnts(lab):
    return np.array([(lab[sbj == s] == c).sum() for s, c in key]) / ns_
am = cnts(Po.argmax(1)); lc = cnts(lab0); q7 = cnts(D["Q7o"].argmax(1)); q9 = cnts(D["Q9o"].argmax(1))
b2 = cnts(D["Bpo"].argmax(1))
print("key        true   pred  P-argmax B-argmax decoded Q7 Q9")
for i in np.flatnonzero(~((true > 55) & (true < 150)) | (np.abs(cnt - true) > 25)):
    print(f"{tuple(key[i])!s:9s} {true[i]:6.1f} {cnt[i]:6.1f} {am[i]:7.1f} {b2[i]:7.1f} {lc[i]:7.1f} {q7[i]:6.1f} {q9[i]:6.1f}")
print(f"MAE: pred {np.abs(cnt-true).mean():.2f} P-argmax {np.abs(am-true).mean():.2f} B-argmax {np.abs(b2-true).mean():.2f} decoded {np.abs(lc-true).mean():.2f} Q7 {np.abs(q7-true).mean():.2f} Q9 {np.abs(q9-true).mean():.2f}")
# label-free iteration: target <- (1-a) * pred + a * decoded label counts
for a in (0.25, 0.5, 0.75, 1.0):
    c = cnt.copy()
    for it in range(3):
        f, lab = C.f1_from_counts(D, Po, key, c); dc = cnts(lab)
        print(f"  a={a} iter {it}: target err {np.abs(c-true).mean():.2f} F1 {f:.4f} decoded err {np.abs(dc-true).mean():.2f}")
        c = (1 - a) * cnt + a * dc
