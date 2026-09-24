"""Probe kernel: verify competition data layout on Kaggle and time a few operations."""
import os, time, glob
import numpy as np, pandas as pd
t0 = time.time()
print("/kaggle/input:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "MISSING")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count("/") - 2
    if depth <= 3:
        print(root, "dirs:", dirs[:10], "files:", [(f, os.path.getsize(os.path.join(root, f)) // 1000000) for f in sorted(files)[:12]])
print("cpus", os.cpu_count())
cands = glob.glob("/kaggle/input/*/test/test_meta_data.csv") + glob.glob("/kaggle/input/*/*/test/test_meta_data.csv")
print("candidates:", cands)
ROOT = os.path.dirname(os.path.dirname(cands[0])) if cands else "/kaggle/input/3rd-wear-dataset-challenge-hasca-2026"
print("ROOT =", ROOT)
tm = pd.read_csv(f"{ROOT}/test/test_meta_data.csv"); print(tm.head(), tm.shape)
xi = np.load(f"{ROOT}/test/test_inertial_data.npy", mmap_mode="r"); print("test inertial", xi.shape, xi.dtype)
xv = np.load(f"{ROOT}/test/test_videomae_data.npy", mmap_mode="r"); print("test video", xv.shape, xv.dtype)
print("video row0 frame stats", np.asarray(xv[0]).reshape(xv.shape[1], -1).mean(), np.asarray(xv[0]).std())
df = pd.read_csv(f"{ROOT}/train/inertial_feat/sbj_0.csv"); print(df.head(3)); print(df.shape); print(df.label.value_counts())
v = np.load(f"{ROOT}/train/videomae_feat/sbj_0.npy", mmap_mode="r"); print("train video sbj_0", v.shape, v.dtype, "frames/sec-of-imu", v.shape[0] / (len(df) / 50))
os.makedirs("/kaggle/working/out", exist_ok=True)
pd.DataFrame({"k": ["n_test", "n_sbj0_rows", "n_sbj0_frames"], "v": [len(tm), len(df), v.shape[0]]}).to_csv("/kaggle/working/out/probe.csv", index=False)
print("elapsed", time.time() - t0)
