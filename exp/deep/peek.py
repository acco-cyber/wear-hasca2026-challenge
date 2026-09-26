import numpy as np, pandas as pd, sys
sys.path.insert(0, r"E:\Claude code\wear\exp\base")
from common import *
m = meta()
print(m.head()); print(m.dtypes); print(m.session.unique()[:30])
imu = np.load(r"E:\Claude code\wear\data\prep\train_imu.npy", mmap_mode="r")
print(imu.shape, imu.dtype)
x = np.asarray(imu[:5000]).astype(np.float32)
print("train imu stats per limb", np.nanmean(x, axis=(0, 2)), np.nanstd(x, axis=(0, 2)))
te = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy")
tm = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
print(tm.head()); print(tm.sensor_location.value_counts())
print("test imu", te.shape, te.mean(axis=(0, 1)), te.std(axis=(0, 1)))
for l in LIMBS:
    s = tm.sensor_location.values == l
    print(l, te[s].mean(axis=(0, 1)), te[s].std(axis=(0, 1)))
xt = np.asarray(imu).astype(np.float32)
for i, l in enumerate(LIMBS):
    print("train", l, np.nanmean(xt[:, i], axis=(0, 1)), np.nanstd(xt[:, i], axis=(0, 1)))
v = np.load(r"E:\Claude code\wear\data\prep\train_vid_pca.npy", mmap_mode="r"); print(v.shape, v.dtype)
vt = np.load(r"E:\Claude code\wear\data\prep\test_vid_pca.npy", mmap_mode="r"); print(vt.shape, vt.dtype)
print("vid std train", np.asarray(v[:3000]).astype(np.float32).std(axis=(0, 1))[:8])
print("vid std test", np.asarray(vt[:3000]).astype(np.float32).std(axis=(0, 1))[:8])
fold = sec_fold()
print("fold0 subjects", np.unique(m.sbj[fold == 0]), (fold == 0).sum())
ref = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy"))
print("lgbm_v1 full", eval_single(ref, name="v1"))
o = ref.copy(); o[fold != 0] = np.nan
# fold-0-only eval
rl = single_limb_pick(np.asarray(ref[:, :, :1]), 5)
y = m.y.to_numpy(); pur = m.pur.to_numpy()
idx = np.where((rl >= 0) & (pur >= 0.8) & (fold == 0))[0]
from sklearn.metrics import f1_score
print("v1 fold0 single", f1_score(y[idx], ref[idx, rl[idx]].argmax(1), average="macro"), len(idx))
import glob
for p in [r"E:\Claude code\wear\exp\base\v3b\oof.npy", r"E:\Claude code\wear\work\fusion_v1\oof.npy"]:
    try:
        r = np.load(p); P = r[idx, rl[idx]] if r.ndim == 3 else r[idx]
        P = np.where(np.isnan(P[:, :1]), 1 / 19, P)
        print(p, r.shape, "fold0 single", f1_score(y[idx], P.argmax(1), average="macro"))
    except Exception as e:
        print(p, e)
