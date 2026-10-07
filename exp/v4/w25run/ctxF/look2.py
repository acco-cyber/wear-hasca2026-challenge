import numpy as np, os
z = np.load(r"E:\Claude code\wear\work\w25\w25.npz", allow_pickle=True)
m = np.load(r"E:\Claude code\wear\work\w25\match.npz", allow_pickle=True)
X = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy").astype(np.float32)
st = np.load(r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4\stage.npz", allow_pickle=True)
acc, sb, lb = z["acc"], z["sbj"], z["limb"]
tw, corr, mg = m["twin"], m["corr"], m["margin"]
d = np.abs(acc[tw] - X).max((1, 2)); ex = d < 1e-4
sens = st["sensor_test"]; tsb = st["test_sbj"]
for s in (22, 23, 24, 25):
    print(s, " ".join(f"L{L}:{ex[(tsb == s) & (sens == L)].mean():.3f}({((tsb == s) & (sens == L)).sum()})" for L in range(4)))
# non-exact: relation of our tile to its twin: scale / mean / per-axis
ne = np.flatnonzero(~ex)[:4000]
A = X[ne]; B = acc[tw[ne]]
print("mean our", A.mean((0, 1)), "mean twin", B.mean((0, 1)))
print("std our", A.std((0, 1)), "std twin", B.std((0, 1)))
# per tile linear regression twin ~ a*our + b per axis
for k in range(3):
    a_ = ((A[:, :, k] - A[:, :, k].mean(1, keepdims=True)) * (B[:, :, k] - B[:, :, k].mean(1, keepdims=True))).sum(1) / (((A[:, :, k] - A[:, :, k].mean(1, keepdims=True)) ** 2).sum(1) + 1e-9)
    print("axis", k, "slope quantiles", np.quantile(a_, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3), "mean shift q", np.quantile(B[:, :, k].mean(1) - A[:, :, k].mean(1), [0.1, 0.5, 0.9]).round(3))
r = np.abs(B - A).mean((1, 2)); print("mean abs diff q", np.quantile(r, [0.1, 0.5, 0.9]))
# is the residual noise-like? corr of residual with lag-1 itself
R = B - A; ac = (R[:, 1:] * R[:, :-1]).sum((1, 2)) / ((R ** 2).sum((1, 2)) + 1e-9); print("residual lag1 autocorr q", np.quantile(ac, [0.1, 0.5, 0.9]))
# other limbs: rows of 2025 not twins: smoothness proxy (2nd diff power / power) vs twin rows
twset = np.zeros(len(acc), bool); twset[tw[ex]] = True
def rough(T):
    return (np.diff(T, 2, axis=1) ** 2).mean((1, 2)) / (np.diff(T, 1, axis=1) ** 2).mean((1, 2)).clip(1e-9)
for L25 in range(4):
    mm = lb == L25
    print("2025 limb", L25, "rough exact-twin rows", np.quantile(rough(acc[mm & twset]), [0.25, 0.5, 0.75, 0.95]).round(3), "other rows", np.quantile(rough(acc[mm & ~twset]), [0.25, 0.5, 0.75, 0.95]).round(3))
