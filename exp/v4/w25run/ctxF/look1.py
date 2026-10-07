import numpy as np, os
z = np.load(r"E:\Claude code\wear\work\w25\w25.npz", allow_pickle=True)
m = np.load(r"E:\Claude code\wear\work\w25\match.npz", allow_pickle=True)
X = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy")
st = np.load(r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4\stage.npz", allow_pickle=True)
acc, sb, lb = z["acc"], z["sbj"], z["limb"]
tw, corr, mg = m["twin"], m["corr"], m["margin"]
print("ids == arange", (z["id"] == np.arange(len(acc))).all())
print("pipeline ids == arange", (st["ids"].astype(int) == np.arange(len(X))).all())
for s in np.unique(sb):
    print("sbj", s, [int(((sb == s) & (lb == L)).sum()) for L in range(4)], "our tiles", int((st["test_sbj"] == s).sum()))
d = np.abs(acc[tw] - X.astype(np.float32)).max((1, 2))
print("max abs diff quantiles", np.quantile(d, [0.5, 0.8, 0.83, 0.84, 0.85, 0.9, 0.95, 0.99]))
ex = d < 1e-4
print("exact share", ex.mean(), "twin sbj ok", (sb[tw] == st["test_sbj"]).mean())
sens = st["sensor_test"]
for L in range(4):
    print("pipeline sensor", L, "-> 2025 limb counts (exact)", np.bincount(lb[tw[ex & (sens == L)]], minlength=4), " (all)", np.bincount(lb[tw[sens == L]], minlength=4))
print("corr quantiles non-exact", np.quantile(corr[~ex], [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]))
print("margin quantiles non-exact", np.quantile(mg[~ex], [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]))
print("margin quantiles exact", np.quantile(mg[ex], [0, 0.05, 0.1, 0.25, 0.5]))
u, c = np.unique(tw, return_counts=True); print("duplicate twins", (c > 1).sum(), "dup among exact", len(tw[ex]) - len(np.unique(tw[ex])))
print("scale our", X.std(), "w25", acc.std())
print("nan our", np.isnan(X).sum(), "nan w25", np.isnan(acc).sum())
