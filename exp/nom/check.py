import time, numpy as np, pandas as pd
t0 = time.time()
D = r"E:\Claude code\wear\data"
meta = pd.read_csv(D + r"\prep\train_meta.csv")
print(meta.columns.tolist(), len(meta)); print(meta.head(3))
print(meta.groupby('session').sbj.first().to_dict())
imu = np.load(D + r"\prep\train_imu.npy", mmap_mode='r'); print('imu', imu.shape, imu.dtype)
vp = np.load(D + r"\prep\train_vid_pca.npy", mmap_mode='r'); print('vp', vp.shape, vp.dtype)
comp = np.load(D + r"\prep\pca_components.npy"); mu = np.load(D + r"\prep\pca_mean.npy"); print(comp.shape, mu.shape, comp.dtype)
tp = np.load(D + r"\prep\test_vid_pca.npy", mmap_mode='r'); print('tp', tp.shape)
tv = np.load(D + r"\test\test_videomae_data.npy", mmap_mode='r'); print('tv', tv.shape)
raw = np.asarray(tv[:50], dtype=np.float32).transpose(0, 2, 1)  # (50,15,768)
p = np.asarray(tp[:50], dtype=np.float32)
proj = (raw - mu) @ comp.T
print('proj err', np.abs(proj - p).max(), np.abs(p).max())
rec = p @ comp + mu
print('rec rel err', np.linalg.norm(rec - raw) / np.linalg.norm(raw), 'raw norm per frame', np.linalg.norm(raw, axis=-1).mean())
tm = pd.read_csv(D + r"\test\test_meta_data.csv"); print(tm.head()); print(tm.sensor_location.value_counts())
print('pur stats', meta.pur.describe())
print('nan limbs', np.isnan(np.asarray(imu[::50], dtype=np.float32)).any(axis=(2, 3)).mean(0))
print('time', time.time() - t0)
