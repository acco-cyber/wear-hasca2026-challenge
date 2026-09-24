"""Task (6): other exploitable structure in the test set: id permutation structure, per-limb gravity orientation,
inertial per-subject distributions, session-duration arithmetic, limb assignment randomness."""
import numpy as np, pandas as pd, os
from scipy import stats
D = r"E:\Claude code\wear\data\test"
A = r"E:\Claude code\wear\research\artifacts"
meta = pd.read_csv(os.path.join(D, "test_meta_data.csv"))
X = np.load(os.path.join(D, "test_inertial_data.npy"))
N = len(meta)
# --- session arithmetic
sess = {22: [40 * 60 + 30, 40 * 60 + 10, 2 * 60 + 57], 23: [24 * 60, 26 * 60 + 21, 60 + 47], 24: [17 * 60 + 25, 15 * 60 + 50], 25: [16 * 60 + 5, 15 * 60 + 49]}
for s, d in sess.items():
    print(f"sbj {s}: sessions {d} sum {sum(d)} vs windows {(meta.sbj_id == s).sum()}  diff {(meta.sbj_id == s).sum() - sum(d)}")
# --- id permutation: subject sequence runs test (is the id order a uniform shuffle?)
sb = meta.sbj_id.values
runs = 1 + (sb[1:] != sb[:-1]).sum()
p = meta.sbj_id.value_counts(normalize=True)
exp_runs = 1 + (N - 1) * (1 - (p ** 2).sum())
print(f"subject-sequence runs: observed {runs}, expected under uniform shuffle {exp_runs:.0f}")
lb = meta.sensor_location.values
runs_l = 1 + (lb[1:] != lb[:-1]).sum(); pl = meta.sensor_location.value_counts(normalize=True)
print(f"limb-sequence runs: observed {runs_l}, expected {1 + (N-1)*(1-(pl**2).sum()):.0f}")
# chi-square: limb independent of subject?
ct = pd.crosstab(meta.sbj_id, meta.sensor_location)
chi2, pv, dof, _ = stats.chi2_contingency(ct)
print(f"limb x subject chi2 p={pv:.3f} (limb choice looks uniform-random per window)")
# binomial check of limb counts per subject
for s in sess:
    c = ct.loc[s].values; n = c.sum()
    print(f"  sbj {s} limb counts {c.tolist()} ; expected {n/4:.0f} each; chi2 p={stats.chisquare(c).pvalue:.3f}")
# --- gravity orientation per subject x limb (mean acc vector) : sensor mounting consistency
print("\nmean acc vector per subject x limb (gravity direction, g units):")
rows = []
for s in sess:
    for l in sorted(meta.sensor_location.unique()):
        m = (meta.sbj_id == s) & (meta.sensor_location == l)
        v = X[m.values].mean(axis=(0, 1)); nrm = np.linalg.norm(X[m.values], axis=2)
        rows.append(dict(sbj=s, limb=l, n=m.sum(), mx=v[0], my=v[1], mz=v[2], mean_norm=nrm.mean(),
                         frac_windows_static=(X[m.values].std(axis=(1, 2)) < 0.05).mean(),
                         std_of_window_std_median=np.median(X[m.values].std(axis=1).mean(1))))
df = pd.DataFrame(rows); print(df.round(3).to_string()); df.to_csv(os.path.join(A, "test_inertial_limb_stats.csv"), index=False)
# --- activity-level proxy: window energy distribution per subject (fraction of low-motion windows ~ null/stretch)
print("\nper-subject window motion (std over samples, mean over axes) quantiles:")
for s in sess:
    m = (meta.sbj_id == s).values; e = X[m].std(axis=1).mean(1)
    print(f"  sbj {s}: p10 {np.percentile(e,10):.3f} p25 {np.percentile(e,25):.3f} p50 {np.percentile(e,50):.3f} p75 {np.percentile(e,75):.3f} p90 {np.percentile(e,90):.3f}; frac<0.05 {(e<0.05).mean():.3f} frac<0.1 {(e<0.1).mean():.3f}")
# --- are there windows where the inertial signal is a pure constant (sensor dropout)?
flat = (X.std(axis=1).max(1) < 1e-4)
print("windows with an axis-constant signal:", flat.sum())
# --- sample-level: do first/last samples look like a window boundary artefact (e.g., filtering edge effects)? compare |diff| at edges vs middle
dd = np.abs(np.diff(X, axis=1)).mean(axis=(0, 2))
print("mean |diff| by sample position: first3", dd[:3].round(4), "middle", dd[23:26].round(4), "last3", dd[-3:].round(4))
print("done")
