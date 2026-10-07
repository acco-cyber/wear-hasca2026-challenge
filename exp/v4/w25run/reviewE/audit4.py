"""Review E, part 4: verify every non-exact anchor of deaug.py with the FORWARD augmentation model (b = -Warp(x) for
left_arm, b = x A + N0 for right_arm): a wrong twin would show a large forward residual. Also: rows used twice, and
whether the anchored set is consistent with the chain caches."""
import os, csv
import numpy as np
W = r"E:\Claude code\wear"; TD = os.path.join(W, "exp", "v4", "w25run", "testD")
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; L25 = ["left_arm", "left_leg", "right_arm", "right_leg"]; MAP25 = np.array([SENS.index(x) for x in L25])
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
meta = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in meta]); se26 = np.array([SENS.index(r["sensor_location"]) for r in meta])
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; pl25 = MAP25[z["limb"]]
d = np.load(os.path.join(TD, "cache", "deaug_inputs.npz")); tw = d["twin"]; how = d["how"]; an = d["anch"]
Wf, Aq, N0 = d["Wf"], d["A_ra"], d["N0"]
print("how counts (0 none, 1 exact, 2 LA, 3 RA):", np.bincount(how, minlength=4).tolist())
assert (an == (how > 0)).all() and (tw[an] >= 0).all() and len(np.unique(tw[an])) == an.sum()
assert (s25[tw[an]] == s26[an]).all() and (pl25[tw[an]] == se26[an]).all()
for code, nm in ((2, "LA"), (3, "RA")):
    q = np.flatnonzero(how == code); X = A26[q]; B = A25[tw[q]]
    if code == 2:
        P = -np.einsum("kt,nkc->ntc", Wf, X)            # Bp ~ Xp @ Wf with per-axis rows: b[t] = -sum_k x[k] Wf[k, t]
    else:
        P = X @ Aq + N0[None]
    r = np.abs(P - B).max((1, 2))
    # nearest alternative 2025 row of the same subject/limb under the forward model (margin)
    alt = []
    for i, p in zip(q[:300], P[:300]):
        rr = np.flatnonzero((s25 == s26[i]) & (pl25 == se26[i])); dd = np.abs(A25[rr] - p[None]).max((1, 2)); dd.sort(); alt.append(dd[1])
    print(f"{nm}: {len(q)} anchors; forward-model max|resid| q50 {np.quantile(r, .5):.2e} q99 {np.quantile(r, .99):.2e} max {r.max():.2e}; "
          f"second-best row (first 300) min {np.min(alt):.3f} q01 {np.quantile(alt, .01):.3f}")
# type-ii right-arm tiles: no row of their subject fits the forward model
q = np.flatnonzero((how == 0) & (se26 == 0))
print(f"unanchored right_arm tiles {len(q)}; unanchored other limbs {np.sum((how == 0) & (se26 != 0))}")
