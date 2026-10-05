"""Per-subject context: true / predicted null fraction, subject-level features, flips of subject 10."""
import os
import numpy as np
from ne_common import HERE, K7
r = np.load(os.path.join(HERE, "res_null_cand3.npz")); z = np.load(os.path.join(HERE, "feats.npz")); b = np.load(os.path.join(HERE, "base_cache.npz"))
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
y = st["oof_y"]; sbj = st["oof_sbj"]; lab = b["ref_o"]; names = [str(s) for s in z["names"]]; Xo = z["Xo"]
tsb = st["test_sbj"]; lt = b["ref_t"]
for s in np.unique(sbj):
    ii = sbj == s
    print(f"sbj {s:2d}: n {ii.sum():5d} true null {(y[ii] == 0).mean():.3f} pred null {(lab[ii] == 0).mean():.3f} acc {(lab[ii] == y[ii]).mean():.3f} "
          f"P0mean {Xo[ii, names.index('sbj_P0_mean')][0]:.3f} sens {np.bincount(st['sensor_oof'][ii], minlength=4)}")
for s in np.unique(tsb):
    ii = tsb == s
    print(f"test {s}: n {ii.sum()} pred null {(lt[ii] == 0).mean():.3f} sens {np.bincount(st['sensor_test'][ii], minlength=4)}")
idx, p = r["idx"], r["p_outer"]
m = (sbj[idx] == 10) & (lab[idx] == 0) & (p < 0.4)
g = idx[m]
print("sbj10 flips: true label hist", np.bincount(y[g], minlength=19))
for nm in ["P_lo", "Q_lo", "B_lo", "W_lo", "TA7_lo", "S37_lo", "nbP_sum3", "nbnull_sum3", "chg_min", "ener", "vmot", "sbj_null_frac", "lab_frac"]:
    j = names.index(nm); oth = (lab[idx] == 0) & (sbj[idx] != 10) & (p < 0.4)
    print(f"{nm:14s} sbj10 flips {np.nanmean(Xo[g, j]):8.3f}  other flips {np.nanmean(Xo[idx[oth], j]):8.3f}  all null cand {np.nanmean(Xo[idx[lab[idx] == 0], j]):8.3f}")
