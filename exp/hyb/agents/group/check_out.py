import numpy as np, pandas as pd
z = np.load(r"E:\Claude code\wear\exp\hyb\agents\group\test_groups.npz", allow_pickle=True)
print({k: (z[k].shape, z[k].dtype) for k in z.files})
w = np.load(r"E:\Claude code\wear\work\w25\w25.npz"); m = np.load(r"E:\Claude code\wear\work\w25\match.npz")
meta = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
rows, conf, raw, anc = z["w25_rows"], z["conf"], z["conf_raw"], z["anchor_limb"]
T = len(rows)
assert (rows[np.arange(T), anc] == m["twin"]).all(), "own slot != twin"
ok = rows >= 0
# every assigned row has the right subject and limb
assert (w["sbj"][rows[ok]] == np.repeat(meta.sbj_id.to_numpy()[:, None], 4, 1)[ok]).all()
assert (w["limb"][rows[ok]] == np.tile(np.arange(4), (T, 1))[ok]).all()
print("own-slot==twin OK; subject/limb of assigned rows OK; conf of unassigned slots all zero:", (conf[~ok] == 0).all())
print("conf of assigned non-anchor slots: min %.3f mean %.3f" % (conf[ok & (conf < 1)].min(), conf[ok & (conf < 1)].mean()))
full = z["w25_rows_unthresholded"]
for thr in (0.6, 0.7, 0.8, 0.85, 0.9):
    keep = (raw >= thr) & (full >= 0); keep[np.arange(T), anc] = True
    n = keep.sum(1); frac = np.bincount(n, minlength=5)[1:5] / T
    print(f"thr {thr:.2f}: tiles with 1/2/3/4 limbs {np.round(frac, 3).tolist()}; non-anchor slots assigned {(keep.sum() - T)} "
          f"(sym-partner slots {sum(keep[i, {0:2,2:0,1:3,3:1}[anc[i]]] for i in range(T))})")
