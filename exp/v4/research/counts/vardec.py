"""Variance structure of the true counts (regular keys) and of the baseline errors (diagnostic)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import N_CLS

D = C.load()
X, key, Xt, kt, true, kf = C.build(D, [D["Po"], D["Bpo"]], [D["Pt"], D["Bpt"]])
cnt = np.load(os.path.join(C.HERE, "diag_cnt_F0_ridge.npy"))
reg = (true > 55) & (true < 150)
subs = np.unique(key[:, 0])
# two-way additive fit on regular keys (oracle: uses all labels)
A = np.zeros((len(true), len(subs) + N_CLS - 1))
for i, (s, c) in enumerate(key):
    A[i, np.searchsorted(subs, s)] = 1; A[i, len(subs) + c - 1] = 1
coef, *_ = np.linalg.lstsq(A[reg], true[reg], rcond=None); fit = A @ coef
print(f"regular keys: sd {true[reg].std():.2f}; MAE of const {np.abs(true[reg] - true[reg].mean()).mean():.2f}; "
      f"class-mean MAE {np.mean([np.abs(true[reg & (key[:,1]==c)] - true[reg & (key[:,1]==c)].mean()).mean() for c in range(1, N_CLS)]):.2f}; "
      f"additive subject+class (oracle) MAE {np.abs(true[reg] - fit[reg]).mean():.2f}; model MAE {np.abs(true[reg] - cnt[reg]).mean():.2f}")
# per subject within-sd
for s in subs:
    m = (key[:, 0] == s) & reg
    print(f"  sbj {s:2d}: within-subject sd {true[m].std():5.1f}  resid after class effect {np.std(true[m] - coef[len(subs) + key[m, 1] - 1]):5.1f}  model err {np.abs(cnt[m]-true[m]).mean():5.2f}")
# error correlation between classes within subject? residual of model vs residual of additive
r_model = cnt - true
print("corr(model error, true - additive fit) on regular:", np.corrcoef(r_model[reg], (true - fit)[reg])[0, 1].round(3))
