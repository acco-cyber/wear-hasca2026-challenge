"""Task (1) shapes/dtypes, per-subject/per-limb counts; (4) inertial boundary matching;
(2c) inertial continuity for consecutive ids with same limb; (6) exact duplicate inertial rows."""
import numpy as np, pandas as pd, os, json, time
from collections import Counter

D = r"E:\Claude code\wear\data\test"
A = r"E:\Claude code\wear\research\artifacts"
os.makedirs(A, exist_ok=True)

meta = pd.read_csv(os.path.join(D, "test_meta_data.csv"))
X = np.load(os.path.join(D, "test_inertial_data.npy"), mmap_mode="r")
print("meta shape", meta.shape, "cols", list(meta.columns))
print("inertial shape", X.shape, "dtype", X.dtype)
print("ids monotonic 0..N-1:", (meta["id"].values == np.arange(len(meta))).all())
print("\nper-subject counts:\n", meta["sbj_id"].value_counts().sort_index())
print("\nper-limb counts:\n", meta["sensor_location"].value_counts())
print("\nper-subject x limb:\n", pd.crosstab(meta["sbj_id"], meta["sensor_location"]))

X = np.asarray(X, dtype=np.float64)
print("\ninertial global stats: min %.4f max %.4f mean %.4f std %.4f" % (X.min(), X.max(), X.mean(), X.std()))
print("any NaN:", np.isnan(X).any())
# per-axis stats
for a in range(3):
    print(f"  axis{a}: mean {X[:,:,a].mean():.4f} std {X[:,:,a].std():.4f}")
# value quantization: are values on a grid?
vals = np.unique(X[:2000].ravel())
print("unique vals in first 2000 windows:", len(vals), "sample", vals[:10])
diffs = np.diff(np.sort(np.unique(np.round(X[:200].ravel(), 8))))
print("min positive diff between distinct values:", diffs[diffs > 0].min() if (diffs > 0).any() else None)

# ---- (2c) consecutive-id inertial continuity (same subject, same limb)
rows = []
for s in sorted(meta.sbj_id.unique()):
    idx = meta.index[meta.sbj_id == s].values
    limbs = meta.sensor_location.values[idx]
    same = [(i, j) for i, j in zip(idx[:-1], idx[1:]) if meta.sensor_location[i] == meta.sensor_location[j]]
    if not same:
        continue
    i = np.array([p[0] for p in same]); j = np.array([p[1] for p in same])
    d_consec = np.linalg.norm(X[i, -1, :] - X[j, 0, :], axis=1)   # last sample of i vs first of j
    # within-window step size as reference
    d_intra = np.linalg.norm(np.diff(X[idx], axis=1), axis=2).ravel()
    # random same-subject same-limb pairs
    rng = np.random.default_rng(0)
    rp = []
    for lb in np.unique(limbs):
        li = idx[limbs == lb]
        if len(li) < 2: continue
        a = rng.choice(li, 5000); b = rng.choice(li, 5000)
        m = a != b
        rp.append(np.linalg.norm(X[a[m], -1, :] - X[b[m], 0, :], axis=1))
    rp = np.concatenate(rp)
    rows.append(dict(sbj=s, n_pairs_consec_same_limb=len(same),
                     consec_median=np.median(d_consec), consec_mean=d_consec.mean(),
                     random_median=np.median(rp), random_mean=rp.mean(),
                     intra_step_median=np.median(d_intra), intra_step_mean=d_intra.mean(),
                     frac_consec_lt_1e6=(d_consec < 1e-6).mean(),
                     frac_consec_lt_intra_median=(d_consec < np.median(d_intra)).mean(),
                     frac_random_lt_intra_median=(rp < np.median(d_intra)).mean()))
df = pd.DataFrame(rows)
print("\n(2c) inertial continuity consecutive ids (same subject & limb) vs random:")
print(df.to_string())
df.to_csv(os.path.join(A, "inertial_consec_continuity.csv"), index=False)

# ---- (4) boundary matching: any window's first sample == another window's last sample (same sbj, same limb)
print("\n(4) boundary matching first[j] == last[i] within 1e-6 (same subject, same limb)")
res = []
for s in sorted(meta.sbj_id.unique()):
    for lb in sorted(meta.sensor_location.unique()):
        idx = meta.index[(meta.sbj_id == s) & (meta.sensor_location == lb)].values
        if len(idx) < 2: continue
        first = X[idx, 0, :]; last = X[idx, -1, :]
        # exact-ish match via rounding to 1e-6 grid
        key_first = {tuple(np.round(v, 6)): [] for v in first}
        for k, v in zip(map(lambda v: tuple(np.round(v, 6)), first), idx): key_first[k].append(v)
        n_match = 0; n_amb = 0; pairs = []
        for i_, v in zip(idx, last):
            k = tuple(np.round(v, 6))
            if k in key_first:
                cands = [c for c in key_first[k] if c != i_]
                if cands:
                    n_match += 1
                    if len(cands) > 1: n_amb += 1
                    pairs.append((i_, cands[0]))
        # also nearest-neighbour distance (min over j != i) of last[i] to first[j]
        from scipy.spatial import cKDTree
        tree = cKDTree(first)
        dd, jj = tree.query(last, k=2)
        # exclude self
        nn = np.where(idx[jj[:, 0]] == idx, dd[:, 1], dd[:, 0])
        res.append(dict(sbj=s, limb=lb, n=len(idx), n_exact_boundary_match=n_match, n_ambiguous=n_amb,
                        nn_dist_median=np.median(nn), nn_dist_p10=np.percentile(nn, 10), nn_dist_p90=np.percentile(nn, 90),
                        frac_nn_lt_1e3=(nn < 1e-3).mean()))
        # first sample equals last sample of self? (sanity)
df4 = pd.DataFrame(res)
print(df4.to_string())
df4.to_csv(os.path.join(A, "inertial_boundary_match.csv"), index=False)

# ---- (6) exact duplicate inertial windows (across all)
h = {}
dup = 0; dup_pairs = []
for i in range(len(X)):
    k = X[i].tobytes()
    if k in h:
        dup += 1; dup_pairs.append((h[k], i))
    else:
        h[k] = i
print("\n(6) exact duplicate inertial windows:", dup)
if dup_pairs:
    dp = pd.DataFrame(dup_pairs, columns=["i", "j"])
    dp["sbj_i"] = meta.sbj_id.values[dp.i]; dp["sbj_j"] = meta.sbj_id.values[dp.j]
    dp["limb_i"] = meta.sensor_location.values[dp.i]; dp["limb_j"] = meta.sensor_location.values[dp.j]
    print(dp.head(20).to_string())
    dp.to_csv(os.path.join(A, "inertial_exact_dups.csv"), index=False)
# all-zero / constant windows
std_w = X.std(axis=(1, 2))
print("windows with std<1e-6:", (std_w < 1e-6).sum(), "; std<0.01:", (std_w < 0.01).sum())
# gravity check: mean norm per window
nrm = np.linalg.norm(X, axis=2).mean(axis=1)
print("mean |acc| per window: median %.3f p5 %.3f p95 %.3f" % (np.median(nrm), np.percentile(nrm, 5), np.percentile(nrm, 95)))
# subsequence overlap: does window j's first 25 samples equal window i's last 25 (stride-0.5s)? test with hashing
print("\n(4b) half-window overlap test: last 25 samples of i == first 25 samples of j (same sbj, limb)")
tot = 0
for s in sorted(meta.sbj_id.unique()):
    for lb in sorted(meta.sensor_location.unique()):
        idx = meta.index[(meta.sbj_id == s) & (meta.sensor_location == lb)].values
        hf = {}
        for i_ in idx:
            hf.setdefault(np.round(X[i_, :25, :], 5).tobytes(), []).append(i_)
        m = sum(1 for i_ in idx if np.round(X[i_, 25:, :], 5).tobytes() in hf)
        tot += m
print("  total half-overlap matches:", tot)
print("done")
