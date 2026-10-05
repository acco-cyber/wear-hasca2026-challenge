"""Where does the true-count oracle gain come from? (diagnostic)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import FOLDS, macro_f1, N_CLS

D = C.load(); y, fold, sbj = D["y"], D["fold"], D["sbj"]
X, key, Xt, kt, true, kf = C.build(D, [D["Po"], D["Bpo"]], [D["Pt"], D["Bpt"]])
cnt = np.load(os.path.join(C.HERE, "diag_cnt_F0_ridge.npy"))
f0, lab0 = C.f1_from_counts(D, D["Po"], key, cnt)
irr = ~((true > 55) & (true < 150))
c = cnt.copy(); c[irr] = true[irr]; f, _ = C.f1_from_counts(D, D["Po"], key, c); print(f"base {f0:.4f}; irregular keys true {f:.4f}")
c = cnt.copy(); c[~irr] = true[~irr]; f, _ = C.f1_from_counts(D, D["Po"], key, c); print(f"regular keys true {f:.4f}")
# per subject gain
for s in np.unique(sbj):
    c = cnt.copy(); m = key[:, 0] == s; c[m] = true[m]; f, _ = C.f1_from_counts(D, D["Po"], key, c)
    print(f"  subject {s:2d}: true counts -> F1 {f:.4f} ({f - f0:+.4f}); err {np.abs(cnt[m] - true[m]).mean():.2f}")
# half-way oracle: error halved everywhere
c = cnt + 0.5 * (true - cnt); f, _ = C.f1_from_counts(D, D["Po"], key, c); print(f"error halved: {f:.4f}")
c = cnt + 0.25 * (true - cnt); f, _ = C.f1_from_counts(D, D["Po"], key, c); print(f"error x0.75: {f:.4f}")
# soft-tile counts of fused P vs true
soft = np.array([D["Po"][sbj == s, c_].sum() / C.TRAIN_SETS.get(int(s), 1) for s, c_ in key])
am = np.array([(D["Po"][sbj == s].argmax(1) == c_).sum() / C.TRAIN_SETS.get(int(s), 1) for s, c_ in key])
print(f"err soft-count {np.abs(soft - true).mean():.2f}, argmax-count {np.abs(am - true).mean():.2f}")
qq = np.array([(D["Q7o"][sbj == s].argmax(1) == c_).sum() / C.TRAIN_SETS.get(int(s), 1) for s, c_ in key])
print(f"err kernel Q7 argmax-count {np.abs(qq - true).mean():.2f}")
f1lab = np.array([(lab0[sbj == s] == c_).sum() / C.TRAIN_SETS.get(int(s), 1) for s, c_ in key])
print(f"err of the baseline decode's own label counts {np.abs(f1lab - true).mean():.2f}")
