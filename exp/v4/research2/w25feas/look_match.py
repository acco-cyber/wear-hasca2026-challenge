import os, numpy as np, pandas as pd
W = r"E:\Claude code\wear"
m = np.load(os.path.join(W, "work", "w25", "match.npz"))
corr, margin, twin = m["corr"], m["margin"], m["twin"]
print("corr quantiles", np.quantile(corr, [0.01, 0.05, 0.1, 0.15, 0.2, 0.5]).round(4))
print("corr>0.999", (corr > 0.999).mean().round(4), "corr>0.99", (corr > 0.99).mean().round(4), "margin>0.01", (margin > 0.01).mean().round(4))
z = np.load(os.path.join(W, "work", "w25", "w25.npz"))
A25 = z["acc"]; s25 = z["sbj"]; l25 = z["limb"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
print("A26", A26.shape, A26.dtype, "A25", A25.shape)
tm = pd.read_csv(os.path.join(W, "data", "test", "test_meta_data.csv"))
print(tm.head()); print(tm.sensor_location.value_counts())
# exact equality of twins
d = np.abs(A26.reshape(len(A26), -1)[:, :150] if A26.shape[1] != 50 else A26.reshape(len(A26), -1))
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
a26 = A26 if A26.shape[1] == 50 else np.transpose(A26, (0, 2, 1))
diff = np.abs(a26 - A25[twin]).max((1, 2))
print("max abs diff to twin quantiles", np.quantile(diff, [0.5, 0.8, 0.85, 0.9, 0.95, 0.99]).round(5))
print("share identical (<1e-5)", (diff < 1e-5).mean().round(4))
# rows per subject/limb in 2025
print(pd.crosstab(s25, l25))
print(pd.crosstab(tm.sbj_id, tm.sensor_location))
# static share for non-identical
mag = np.linalg.norm(a26, axis=2); dyn = mag.std(1)
bad = diff >= 1e-5
print("non-identical: dyn std median", np.median(dyn[bad]).round(4), "identical:", np.median(dyn[~bad]).round(4))
# do non-identical ones have an identical tile elsewhere? check scale
r = (np.linalg.norm(A25[twin], axis=2).mean(1) / (mag.mean(1) + 1e-9))
print("scale ratio for non-identical", np.quantile(r[bad], [0.1, 0.5, 0.9]).round(3), "identical", np.quantile(r[~bad], [0.1, 0.5, 0.9]).round(3))
