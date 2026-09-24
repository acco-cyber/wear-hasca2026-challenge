"""Download all 3rd WEAR challenge files into E:/Claude code/wear/data (idempotent)."""
import os, sys, zipfile, time
os.environ.setdefault("KAGGLE_API_TOKEN", sys.argv[1] if len(sys.argv) > 1 else "")
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "3rd-wear-dataset-challenge-hasca-2026"
OUT = r"E:\Claude code\wear\data"

files = ["participant_meta_data.txt", "sample_submission.csv",
         "test/test_meta_data.csv", "test/test_inertial_data.npy", "test/test_videomae_data.npy"]
sbjs = [str(i) for i in range(22)] + ["0_2", "14_2"]
files += [f"train/inertial_feat/sbj_{s}.csv" for s in sbjs]
files += [f"train/videomae_feat/sbj_{s}.npy" for s in sbjs]

api = KaggleApi(); api.authenticate()
t0 = time.time()
for f in files:
    dest_dir = os.path.join(OUT, os.path.dirname(f))
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(OUT, f)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print("skip", f, flush=True); continue
    print(f"[{time.time()-t0:7.0f}s] downloading {f}", flush=True)
    for attempt in range(3):
        try:
            api.competition_download_file(COMP, f, path=dest_dir, quiet=True, force=True)
            break
        except Exception as e:
            print("  retry", attempt, e, flush=True); time.sleep(5)
    leaf = os.path.basename(f)
    z = os.path.join(dest_dir, leaf + ".zip")
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        os.remove(z)
    if not os.path.exists(target):
        # some files come down unzipped under their leaf name
        cand = os.path.join(dest_dir, leaf)
        print("  WARN target missing; dir has:", os.listdir(dest_dir)[:5], flush=True)
print("ALL DONE", time.time() - t0)
tot = 0
for root, _, fs in os.walk(OUT):
    for x in fs:
        tot += os.path.getsize(os.path.join(root, x))
print("total bytes", tot)
