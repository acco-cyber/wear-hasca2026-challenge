"""Kaggle prep kernel: tile every training session into 1-second windows in the exact TEST format
and emit compact artifacts (video frames PCA-compressed) small enough to download over a slow link.

Outputs (/kaggle/working):
  train_meta.csv        one row per second: session, sbj, t, y (majority), y_c (centre), pur (purity), n_nan_limbs
  train_imu.npy         (N,4,50,3) float16, limb order LIMBS = left_arm, left_leg, right_arm, right_leg (NaN kept)
  train_vid_pca.npy     (N,15,K) float16  centre-15 frames [30t+8, 30t+23) projected by PCA(K)
  test_vid_pca.npy      (12234,15,K) float16 same projection (test file transposed from (N,768,15))
  pca_mean.npy (768,), pca_components.npy (K,768) float32, pca_explained.npy
  train_vid_mean768.npy (N,768) float16 mean over the 15 raw frames (for kNN / pooled models)
  test_vid_mean768.npy  (12234,768) float16
"""
import os, glob, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

t0 = time.time()
ROOT = "/kaggle/input/competitions/3rd-wear-dataset-challenge-hasca-2026"
OUT = "/kaggle/working"
K = 160
CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
    "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
    "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
    "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
    "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LAB2ID = {c: i for i, c in enumerate(CLASS_NAMES)}
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
FS, FPS, V0, V1 = 50, 30, 8, 23

sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/train/inertial_feat/sbj_*.csv"))
print("sessions", len(sessions), sessions, flush=True)

# ---- 1. fit PCA on a random frame subsample from every session
rng = np.random.RandomState(0)
samp = []
for s in sessions:
    V = np.load(f"{ROOT}/train/videomae_feat/{s}.npy", mmap_mode="r")
    idx = np.sort(rng.choice(V.shape[0], min(12000, V.shape[0]), replace=False))
    samp.append(np.asarray(V[idx], np.float32))
samp = np.concatenate(samp)
print("pca sample", samp.shape, f"{time.time()-t0:.0f}s", flush=True)
pca = PCA(n_components=K, svd_solver="randomized", random_state=0).fit(samp)
print("explained", pca.explained_variance_ratio_[:10].round(3), pca.explained_variance_ratio_.sum().round(4), f"{time.time()-t0:.0f}s", flush=True)
np.save(f"{OUT}/pca_mean.npy", pca.mean_.astype(np.float32))
np.save(f"{OUT}/pca_components.npy", pca.components_.astype(np.float32))
np.save(f"{OUT}/pca_explained.npy", pca.explained_variance_ratio_.astype(np.float32))
del samp
comps, mean = pca.components_.astype(np.float32), pca.mean_.astype(np.float32)

# ---- 2. tile every session
metas, imus, vids, vmeans = [], [], [], []
for s in sessions:
    df = pd.read_csv(f"{ROOT}/train/inertial_feat/{s}.csv")
    cols = [f"{l}_acc_{a}" for l in LIMBS for a in "xyz"]
    A = df[cols].to_numpy(np.float32)
    lab = df["label"].map(lambda v: LAB2ID.get(v, 0) if isinstance(v, str) else 0).to_numpy(np.int64)
    V = np.load(f"{ROOT}/train/videomae_feat/{s}.npy", mmap_mode="r")
    n = min(len(A) // FS, (V.shape[0] - V1) // FPS + 1)
    imu = A[: n * FS].reshape(n, FS, 4, 3).transpose(0, 2, 1, 3)
    L = lab[: n * FS].reshape(n, FS)
    y = np.array([np.bincount(r, minlength=19).argmax() for r in L], np.int8)
    pur = (L == y[:, None]).mean(1).astype(np.float32)
    y_c = L[:, FS // 2].astype(np.int8)
    idx = (np.arange(n)[:, None] * FPS + np.arange(V0, V1)[None, :]).reshape(-1)
    fr = np.asarray(V[idx], np.float32).reshape(n, V1 - V0, -1)
    vmeans.append(fr.mean(1).astype(np.float16))
    z = (fr.reshape(-1, fr.shape[-1]) - mean) @ comps.T
    vids.append(z.reshape(n, V1 - V0, K).astype(np.float16))
    imus.append(imu.astype(np.float16))
    n_nan = np.isnan(imu).any(axis=(2, 3)).sum(1).astype(np.int8)
    sbj = int(s.split("_")[1])
    metas.append(pd.DataFrame({"session": s, "sbj": sbj, "t": np.arange(n), "y": y, "y_c": y_c, "pur": pur, "n_nan_limbs": n_nan}))
    print(f"{s}: n={n}s samples={len(A)} frames={V.shape[0]} null={np.mean(y==0):.3f} nanlimbs={np.mean(n_nan>0):.3f} {time.time()-t0:.0f}s", flush=True)

meta = pd.concat(metas, ignore_index=True)
meta.to_csv(f"{OUT}/train_meta.csv", index=False)
np.save(f"{OUT}/train_imu.npy", np.concatenate(imus))
np.save(f"{OUT}/train_vid_pca.npy", np.concatenate(vids))
np.save(f"{OUT}/train_vid_mean768.npy", np.concatenate(vmeans))
print("train saved", meta.shape, f"{time.time()-t0:.0f}s", flush=True)
del imus, vids, vmeans

# ---- 3. test video in the same projection
X = np.load(f"{ROOT}/test/test_videomae_data.npy", mmap_mode="r")
outz, outm = [], []
for st in range(0, X.shape[0], 1024):
    b = np.asarray(X[st:st + 1024], np.float32).transpose(0, 2, 1)     # (b,15,768)
    outm.append(b.mean(1).astype(np.float16))
    outz.append(((b.reshape(-1, 768) - mean) @ comps.T).reshape(b.shape[0], 15, K).astype(np.float16))
np.save(f"{OUT}/test_vid_pca.npy", np.concatenate(outz))
np.save(f"{OUT}/test_vid_mean768.npy", np.concatenate(outm))
print("test saved", f"{time.time()-t0:.0f}s", flush=True)
for f in sorted(os.listdir(OUT)):
    print(f"{os.path.getsize(os.path.join(OUT, f))/1e6:8.1f} MB {f}")
print("DONE", time.time() - t0)
