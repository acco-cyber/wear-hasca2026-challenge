import numpy as np, pandas as pd, os
os.environ.setdefault("OMP_NUM_THREADS", "6")
w = np.load(r"E:\Claude code\wear\work\w25\w25.npz", allow_pickle=True)
print("w25 keys", w.files)
for k in w.files:
    a = w[k]
    print(k, a.shape, a.dtype, a[:5] if a.ndim == 1 else "")
m = np.load(r"E:\Claude code\wear\work\w25\match.npz", allow_pickle=True)
print("match keys", m.files)
for k in m.files:
    a = m[k]
    print(k, a.shape, a.dtype, a[:8] if a.ndim == 1 else a[:2])
meta = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
print(meta.shape); print(meta.head()); print(meta.sbj_id.value_counts()); print(meta.sensor_location.value_counts())
x = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy", mmap_mode="r")
print("test npy", x.shape, x.dtype)
# limb counts in w25
sbj = w["sbj"]; limb = w["limb"]
import collections
print(collections.Counter(zip(sbj.tolist(), limb.tolist())))
# check that twin rows match tile samples
twin = m["twin"]
corr = m["corr"] if "corr" in m.files else None
if corr is not None:
    print("corr>0.99:", (corr > 0.99).sum(), "of", len(corr))
# compare tile 0 with its twin
acc = w["acc"]
for t in [0, 1, 2, 100]:
    r = twin[t]
    print(t, r, meta.iloc[t].to_dict(), "sbj/limb of twin", sbj[r], limb[r], "maxabs diff", np.abs(np.asarray(x[t]).astype(np.float32) - acc[r]).max() if x[t].shape == acc[r].shape else (x[t].shape, acc[r].shape))
# nan check on train
df = pd.read_csv(r"E:\Claude code\wear\data\train\inertial_feat\sbj_5.csv")
print(df.shape, df.isna().sum().to_dict())
print(df.label.value_counts(dropna=False).head(30))
