import numpy as np, csv
z = np.load(r"E:\Claude code\wear\work\w25\w25.npz", allow_pickle=True)
for k in z.files:
    v = z[k]; print(k, v.shape, v.dtype, v[:5] if v.ndim == 1 else v.reshape(len(v), -1)[:2, :6])
m = np.load(r"E:\Claude code\wear\work\w25\match.npz", allow_pickle=True)
for k in m.files:
    v = m[k]; print(k, v.shape, v.dtype, v[:8] if v.ndim == 1 else v[:3])
X = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy", mmap_mode="r"); print("test inertial", X.shape, X.dtype)
with open(r"E:\Claude code\wear\data\test\test_meta_data.csv") as f:
    r = csv.reader(f); hdr = next(r); rows = [next(r) for _ in range(5)]
print(hdr); print(rows)
st = np.load(r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4\stage.npz", allow_pickle=True)
for k in st.files:
    v = st[k]; print("stage", k, v.shape, v.dtype)
print(st["ids"][:5], st["test_sbj"][:5], st["sensor_test"][:5])
