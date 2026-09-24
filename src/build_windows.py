"""Build train windows in the exact TEST format.

Per training session (sbj_X.csv + sbj_X.npy):
  second t  ->  IMU (4 limbs x 50 samples x 3 axes), video frames [30t+8, 30t+23) (15 x 768), labels.
Outputs one .npz per session under data/windows/ :
  imu   (n,4,50,3) float32   limb order = LIMBS below (matches test sensor_location names)
  vid   (n,15,768) float16
  y     (n,) int8  majority label of the 50 inertial samples
  y_c   (n,) int8  label at the centre sample
  pur   (n,) float32 purity of the majority label
  t     (n,) int32 second index inside the session
Usage: python build_windows.py [sbj ...]   (default: all sessions present)
"""
import os, sys, glob, time
import numpy as np, pandas as pd

DATA = r"E:\Claude code\wear\data"
OUT = os.path.join(DATA, "windows"); os.makedirs(OUT, exist_ok=True)
CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
    "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
    "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
    "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
    "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LAB2ID = {c: i for i, c in enumerate(CLASS_NAMES)}
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]   # test sensor_location vocabulary (sorted)
FS, FPS, V0, V1 = 50, 30, 8, 23

def build(sbj):
    csv = os.path.join(DATA, "train", "inertial_feat", f"sbj_{sbj}.csv")
    npy = os.path.join(DATA, "train", "videomae_feat", f"sbj_{sbj}.npy")
    out = os.path.join(OUT, f"sbj_{sbj}.npz")
    if os.path.exists(out):
        print("skip", sbj); return
    if not (os.path.exists(csv) and os.path.exists(npy)):
        print("missing inputs for", sbj); return
    t0 = time.time()
    df = pd.read_csv(csv)
    cols = [f"{l}_acc_{a}" for l in LIMBS for a in "xyz"]
    A = df[cols].to_numpy(np.float32)                      # (N,12)
    lab = df["label"].fillna("null").map(LAB2ID)
    if lab.isna().any():
        bad = df["label"][lab.isna()].unique()
        raise ValueError(f"unknown labels in {sbj}: {bad}")
    lab = lab.to_numpy(np.int64)
    V = np.load(npy, mmap_mode="r")                        # (F,768)
    n = min(len(A) // FS, (V.shape[0] - V1) // FPS + 1)
    imu = A[: n * FS].reshape(n, FS, 4, 3).transpose(0, 2, 1, 3).copy()   # (n,4,50,3)
    L = lab[: n * FS].reshape(n, FS)
    y = np.array([np.bincount(r, minlength=19).argmax() for r in L], np.int8)
    pur = (L == y[:, None]).mean(1).astype(np.float32)
    y_c = L[:, FS // 2].astype(np.int8)
    idx = (np.arange(n)[:, None] * FPS + np.arange(V0, V1)[None, :])       # (n,15)
    vid = np.asarray(V[idx.reshape(-1)], np.float16).reshape(n, V1 - V0, V.shape[1])
    np.savez(out, imu=imu, vid=vid, y=y, y_c=y_c, pur=pur, t=np.arange(n, dtype=np.int32))
    print(f"sbj_{sbj}: n={n}s imu={len(A)} frames={V.shape[0]} dtype={V.dtype} "
          f"null={np.mean(y==0):.3f} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(os.path.basename(p)[4:-4] for p in glob.glob(os.path.join(DATA, "train", "inertial_feat", "sbj_*.csv")))
    for s in args:
        build(s)
