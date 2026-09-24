"""Parallel downloader for the large train videomae files (6 threads). Idempotent; coexists with download_data.py."""
import os, sys, zipfile, time, shutil
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("KAGGLE_API_TOKEN", sys.argv[1] if len(sys.argv) > 1 else "")
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "3rd-wear-dataset-challenge-hasca-2026"
OUT = r"E:\Claude code\wear\data"
TMP = os.path.join(OUT, "tmp_par"); os.makedirs(TMP, exist_ok=True)
sbjs = [str(i) for i in range(22)] + ["0_2", "14_2"]
files = [f"train/videomae_feat/sbj_{s}.npy" for s in sbjs][::-1]
files += [f"train/inertial_feat/sbj_{s}.csv" for s in sbjs][::-1]

api = KaggleApi(); api.authenticate()
t0 = time.time()

def one(f):
    target = os.path.join(OUT, f)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return f"skip {f}"
    leaf = os.path.basename(f)
    d = os.path.join(TMP, leaf.replace(".", "_")); os.makedirs(d, exist_ok=True)
    for attempt in range(3):
        try:
            api.competition_download_file(COMP, f, path=d, quiet=True, force=True)
            break
        except Exception as e:
            print("  retry", f, attempt, e, flush=True); time.sleep(5)
    z = os.path.join(d, leaf + ".zip")
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zf: zf.extractall(d)
        os.remove(z)
    src = os.path.join(d, leaf)
    if not os.path.exists(src):
        return f"FAIL {f}: {os.listdir(d)}"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        if os.path.exists(target) and os.path.getsize(target) == os.path.getsize(src):
            os.remove(src); return f"exists {f}"
        os.replace(src, target)
    except PermissionError:
        return f"busy {f} (other downloader writing it), left copy in {src}"
    return f"[{time.time()-t0:6.0f}s] done {f} {os.path.getsize(target)/1e6:.0f} MB"

with ThreadPoolExecutor(6) as ex:
    for msg in ex.map(one, files):
        print(msg, flush=True)
print("ALL DONE", time.time() - t0)
