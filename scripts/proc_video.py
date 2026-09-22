#!/usr/bin/env python3
"""Stream-process train videomae npy v2:
download -> reshape (N*15, 768) -> (N, 15, 768) -> mean/std over 15 frames -> save float16 -> delete npy.
One or more sbj args. Resumable/safe: verifies frame count vs CSV rows."""
import sys, os, subprocess
import numpy as np

COMP = "3rd-wear-dataset-challenge-hasca-2026"
DATA = "/home/z/my-project/data"
OUT = f"{DATA}/train/videomae_pooled"
os.makedirs(OUT, exist_ok=True)
os.makedirs("/tmp/dl_tmp", exist_ok=True)

def download(sbj, npy_path):
    for attempt in range(3):
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMP, "-f",
             f"train/videomae_feat/{sbj}.npy", "-p", "/tmp/dl_tmp", "--force"],
            capture_output=True, text=True, timeout=280,
            env={**os.environ, "PATH": "/home/z/.local/bin:" + os.environ["PATH"],
                 "KAGGLE_API_TOKEN": "KGAT_d6f6367b0cee135abd9dbe0d7cf96388"})
        src_zip = f"/tmp/dl_tmp/{sbj}.npy.zip"
        src = f"/tmp/dl_tmp/{sbj}.npy"
        if os.path.exists(src_zip):
            subprocess.run(["unzip", "-o", "-q", src_zip, "-d", "/tmp/dl_tmp"], check=True)
            os.remove(src_zip)
        if os.path.exists(src):
            os.replace(src, npy_path)
            return True
    return False

def process(sbj):
    out_mean = f"{OUT}/{sbj}_mean.npy"
    out_std = f"{OUT}/{sbj}_std.npy"
    if os.path.exists(out_mean) and os.path.exists(out_std):
        return "SKIP"
    csv_path = f"{DATA}/train/inertial_feat/{sbj}.csv"
    n_rows = sum(1 for _ in open(csv_path)) - 1
    n_win = n_rows // 25
    npy_path = f"{DATA}/train/videomae_feat/{sbj}.npy"
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    if not os.path.exists(npy_path):
        if not download(sbj, npy_path):
            return "DL_FAIL"
    x = np.load(npy_path)
    assert x.ndim == 2 and x.shape[1] == 768, f"{sbj} unexpected {x.shape}"
    N = x.shape[0]
    assert N == n_win * 15, f"{sbj} frames {N} != {n_win}*15"
    xt = x.reshape(n_win, 15, 768)
    m = xt.mean(axis=1).astype(np.float16)
    s = xt.std(axis=1).astype(np.float16)
    np.save(out_mean, m)
    np.save(out_std, s)
    del x, xt
    os.remove(npy_path)
    return f"OK wins={n_win}"

if __name__ == "__main__":
    for sbj in sys.argv[1:]:
        try:
            print(sbj, process(sbj), flush=True)
        except Exception as e:
            print(sbj, "ERR", repr(e)[:200], flush=True)
