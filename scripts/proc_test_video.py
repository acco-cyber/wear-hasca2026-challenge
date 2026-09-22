#!/usr/bin/env python3
"""Process test videomae: download 1.13GB -> (12234,768,15) f64 -> transpose -> mean/std -> save -> delete."""
import os, subprocess
import numpy as np

COMP = "3rd-wear-dataset-challenge-hasca-2026"
DATA = "/home/z/my-project/data"
os.makedirs("/tmp/dl_tmp", exist_ok=True)
out_m, out_s = f"{DATA}/test/vid_mean.npy", f"{DATA}/test/vid_std.npy"
if os.path.exists(out_m) and os.path.exists(out_s):
    print("SKIP"); raise SystemExit

npy_path = f"{DATA}/test/test_videomae_data.npy"
for attempt in range(3):
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-f",
                    "test/test_videomae_data.npy", "-p", "/tmp/dl_tmp", "--force"],
                   capture_output=True, text=True, timeout=540,
                   env={**os.environ, "PATH": "/home/z/.local/bin:" + os.environ["PATH"],
                        "KAGGLE_API_TOKEN": "KGAT_d6f6367b0cee135abd9dbe0d7cf96388"})
    for cand in [f"/tmp/dl_tmp/test_videomae_data.npy.zip", f"/tmp/dl_tmp/test_videomae_data.npy"]:
        if os.path.exists(cand) and cand.endswith(".zip"):
            subprocess.run(["unzip", "-o", "-q", cand, "-d", "/tmp/dl_tmp"], check=True)
            os.remove(cand)
    if os.path.exists(npy_path):
        break
else:
    print("DL_FAIL"); raise SystemExit

x = np.load(npy_path, mmap_mode="r")
print("raw shape:", x.shape, x.dtype, flush=True)
if x.shape[1] == 768:  # (N, 768, T) -> (N, T, 768)
    xt = np.transpose(np.asarray(x, dtype=np.float32), (0, 2, 1))
else:
    xt = np.asarray(x, dtype=np.float32)
print("transposed:", xt.shape, flush=True)
m = xt.mean(axis=1).astype(np.float16)
s = xt.std(axis=1).astype(np.float16)
np.save(out_m, m); np.save(out_s, s)
print("saved:", m.shape, s.shape, flush=True)
del x, xt
os.remove(npy_path)
print("DONE, raw deleted")
