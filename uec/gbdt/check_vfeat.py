"""Compare vfeat.make_feature_matrix with the repo's make_feature_vector on random windows."""
import sys, time
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\uec\gbdt")
from uecstub import tig
import vfeat

rng = np.random.default_rng(0)
Xt = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy", mmap_mode="r")
meta = __import__("pandas").read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 300
idx = rng.choice(len(Xt), n_test, replace=False)
W = np.asarray(Xt[idx], dtype=np.float32)
keys = [tig.normalize_sensor_location(s) for s in meta.sensor_location.to_numpy()[idx]]

# add some train windows (raw csv) incl. edge cases: constant window, zeros
imu = np.load(r"E:\Claude code\wear\data\prep\train_imu.npy", mmap_mode="r")   # (n,4,50,3) float16
j = rng.choice(imu.shape[0], 150, replace=False)
Wtr = np.asarray(imu[j, 0], dtype=np.float32)
Wtr = Wtr[np.isfinite(Wtr).all(axis=(1, 2))]
const = np.ones((2, 50, 3), np.float32) * np.array([0.1, -0.9, 0.3], np.float32)
zeros = np.zeros((1, 50, 3), np.float32)
W = np.concatenate([W, Wtr, const, zeros])
keys = keys + ["la"] * (len(W) - len(keys))

t = time.time()
ref = [tig.make_feature_vector(W[i], keys[i], use_raw_features=False, smoothing_window=5, sensor_embedding_mode="sensor")
       for i in range(len(W))]
t_ref = time.time() - t
names_ref = list(ref[0].keys())
R = np.array([[r[k] for k in names_ref] for r in ref], dtype=np.float64)
t = time.time()
names, V = vfeat.make_feature_matrix(W, np.array(keys))
t_vec = time.time() - t
print(f"n={len(W)} ref {t_ref / len(W) * 1000:.1f} ms/window, vec {t_vec / len(W) * 1000:.3f} ms/window")
print("n features ref/vec", len(names_ref), len(names), "same order:", names_ref == names)
if names_ref != names:
    for a, b in zip(names_ref, names):
        if a != b:
            print("first mismatch", a, b)
            break
    print("missing", set(names_ref) - set(names), "extra", set(names) - set(names_ref))
    sys.exit(1)
absd = np.abs(R - V)
rel = absd / np.maximum(np.abs(R), 1e-6)
bad = (absd > 1e-6) & (rel > 1e-5)
print("max abs diff", absd.max(), "cells off (abs>1e-6 & rel>1e-5):", int(bad.sum()), "of", bad.size)
order = np.argsort(-bad.sum(0))
for c in order[:25]:
    if bad[:, c].sum() == 0:
        break
    r = np.argmax(np.where(bad[:, c], rel[:, c], -1))
    print(f"  {names[c]:45s} n_bad={bad[:, c].sum():4d} worst ref={R[r, c]:.9g} vec={V[r, c]:.9g}")
