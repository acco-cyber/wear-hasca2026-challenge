"""Build the per-tile cache for the UEC-dx2 exp024 CNN8+VideoMAE-aux port (and the XceptionTime port).

One row per 1-s tile of data/prep/train_meta.csv (all 24 sessions, same row order), so the cache lines up
with our standard-protocol OOF layout.  Windows are tile-aligned (start = 50*t), i.e. UEC's window_size=50
windows restricted to stride 50 (the PCA video cache only holds the centre-15 frames [30t+8, 30t+23) of each
tile, which is exactly UEC's center_crop for a start=50t window).

Video: PCA-160 frames reconstructed to 768-d (mean + z @ components), for train AND test, consistently.
UEC's aggregate features are then computed exactly as in train_exp024_cnn8_videomae_aux.py:
vi_mean, vi_std, vi_delta5 (768 each) and the 56 scalar features.

Labels follow UEC's encode_label + assign_window_label_from_array (purity mode, 0.8).
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd

ROOT = r"E:\Claude code\wear\data"; PREP = os.path.join(ROOT, "prep")
OUT = r"E:\Claude code\wear\uec\cnn8\cache"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

LABEL_TO_ID = {"null": 0, "jogging": 1, "jogging (rotating arms)": 2, "jogging (skipping)": 3, "jogging (sidesteps)": 4,
    "jogging (butt-kicks)": 5, "stretching (triceps)": 6, "stretching (lunging)": 7, "stretching (shoulders)": 8,
    "stretching (hamstrings)": 9, "stretching (lumbar rotation)": 10, "push-ups": 11, "push-ups (complex)": 12,
    "sit-ups": 13, "sit-ups (complex)": 14, "burpees": 15, "lunges": 16, "lunges (complex)": 17, "bench-dips": 18}
# UEC SENSOR_COLS order: ra, rl, ll, la  (sensor_keys default = list(SENSOR_COLS))
SENSOR_COLS = {"ra": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"],
               "rl": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
               "ll": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
               "la": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"]}
WS = 50

def encode_labels(col: pd.Series) -> np.ndarray:
    s = col.astype(str).str.strip(); low = s.str.lower()
    out = np.full(len(s), -2, np.int64)
    out[((low == "") | (low == "null")).to_numpy()] = 0
    for name, i in LABEL_TO_ID.items():
        out[(s == name).to_numpy()] = i
    out[((low == "nan") | (low == "none")).to_numpy()] = -1   # None in UEC
    bad = out == -2
    if bad.any():  # numeric labels (not expected)
        out[bad] = s[bad].astype(float).astype(int).to_numpy()
    return out

def row_cos(a, b, eps=1e-6):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps)

def stats_1d(x):  # x (B,L) -> (B,16), UEC _stats_1d
    q10, q25, q50, q75, q90 = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    return np.stack([x.mean(1), x.std(1), x.min(1), x.max(1), x.max(1) - x.min(1), q10, q25, q50, q75, q90, q75 - q25,
                     np.sqrt(np.mean(x ** 2, 1)), np.sum(x ** 2, 1), x[:, 0], x[:, -1], x[:, -1] - x[:, 0]], 1)

def video_feats(fr):  # fr (B,15,768) float32 -> vi_mean, vi_std, vi_delta5, scalar(56)
    fr = np.nan_to_num(fr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vi_mean = fr.mean(1); vi_std = fr.std(1); vi_d5 = fr[:, -5:].mean(1) - fr[:, :5].mean(1)
    first, last = fr[:, 0], fr[:, -1]
    first5, mid5, last5 = fr[:, :5].mean(1), fr[:, 5:10].mean(1), fr[:, 10:].mean(1)
    frame_norm = np.linalg.norm(fr, axis=2)
    diff_norm = np.linalg.norm(np.diff(fr, axis=1), axis=2)
    frame_cos = row_cos(fr[:, :-1], fr[:, 1:])
    sc = np.stack([np.linalg.norm(last - first, axis=1), row_cos(first, last),
                   np.linalg.norm(mid5 - first5, axis=1), np.linalg.norm(last5 - mid5, axis=1), np.linalg.norm(last5 - first5, axis=1),
                   row_cos(first5, mid5), row_cos(mid5, last5), row_cos(first5, last5)], 1)
    scalar = np.concatenate([sc, stats_1d(frame_norm), stats_1d(diff_norm), stats_1d(frame_cos)], 1)
    f = lambda a: np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return f(vi_mean), f(vi_std), f(vi_d5), f(scalar)

meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
N = len(meta)
pmean = np.load(os.path.join(PREP, "pca_mean.npy")).astype(np.float32)
comps = np.load(os.path.join(PREP, "pca_components.npy")).astype(np.float32)
vpca = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
log("meta", meta.shape, "pca", vpca.shape)

imu = np.full((N, 4, WS, 3), np.nan, np.float32)
y_uec = np.full(N, -1, np.int64)       # UEC purity label (-1 = window rejected by label rule)
for sess, g in meta.groupby("session", sort=False):
    df = pd.read_csv(os.path.join(ROOT, "train", "inertial_feat", f"{sess}.csv"), low_memory=False, keep_default_na=False)
    lab = encode_labels(df["label"])
    arrs = [df[c].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) for c in SENSOR_COLS.values()]
    rows = g.index.to_numpy(); ts = g.t.to_numpy()
    for r, t in zip(rows, ts):
        st = WS * int(t)
        if st + WS > len(df):
            continue
        for k, A in enumerate(arrs):
            imu[r, k] = A[st:st + WS]
        w = lab[st:st + WS]
        if (w < 0).any():
            continue
        vals, cnts = np.unique(w, return_counts=True)
        if cnts.max() / WS >= 0.8:
            y_uec[r] = int(vals[cnts.argmax()])
    log(sess, "rows", len(rows), "len", len(df), "labelled", int((y_uec[rows] >= 0).sum()))

fin4 = np.isfinite(imu).all(axis=(2, 3))            # (N,4) per-sensor finite
vid_ok = np.ones(N, bool)
vi_mean = np.zeros((N, 768), np.float32); vi_std = np.zeros((N, 768), np.float32)
vi_d5 = np.zeros((N, 768), np.float32); scal = np.zeros((N, 56), np.float32)
for st in range(0, N, 2048):
    z = np.asarray(vpca[st:st + 2048], np.float32)
    vid_ok[st:st + len(z)] = np.isfinite(z).all(axis=(1, 2))
    fr = pmean + z @ comps
    a, b, c, d = video_feats(fr)
    vi_mean[st:st + len(z)] = a; vi_std[st:st + len(z)] = b; vi_d5[st:st + len(z)] = c; scal[st:st + len(z)] = d
log("train video feats done")

suffix2 = meta.session.str.count("_").to_numpy() >= 2
train_ok = (y_uec >= 0) & fin4.all(1) & vid_ok     # UEC sample rule (all sensors finite, video finite)
log("train_ok", int(train_ok.sum()), "of", N, "| excl suffix2:", int((train_ok & ~suffix2).sum()))
np.save(os.path.join(OUT, "imu.npy"), imu)
np.save(os.path.join(OUT, "y_uec.npy"), y_uec)
np.save(os.path.join(OUT, "train_ok.npy"), train_ok)
np.save(os.path.join(OUT, "suffix2.npy"), suffix2)
np.save(os.path.join(OUT, "vi_mean.npy"), vi_mean); np.save(os.path.join(OUT, "vi_std.npy"), vi_std)
np.save(os.path.join(OUT, "vi_d5.npy"), vi_d5); np.save(os.path.join(OUT, "scalar.npy"), scal)
del vi_mean, vi_std, vi_d5, scal

# ---- test
tv = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r")
M = tv.shape[0]
T = [np.zeros((M, 768), np.float32), np.zeros((M, 768), np.float32), np.zeros((M, 768), np.float32), np.zeros((M, 56), np.float32)]
for st in range(0, M, 2048):
    z = np.asarray(tv[st:st + 2048], np.float32)
    for arr, f in zip(T, video_feats(pmean + z @ comps)):
        arr[st:st + len(z)] = f
for name, arr in zip(["vi_mean", "vi_std", "vi_d5", "scalar"], T):
    np.save(os.path.join(OUT, f"test_{name}.npy"), arr)
log("test video feats done", M)

# sanity: raw test video vs PCA reconstruction for the first 64 rows
raw = np.load(os.path.join(ROOT, "test", "test_videomae_data.npy"), mmap_mode="r")
rr = np.asarray(raw[:64], np.float32).transpose(0, 2, 1)
ra, rb, rc, rd = video_feats(rr)
ta = T[0][:64]
cos = (ra * ta).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(ta, axis=1))
log("raw-vs-recon vi_mean cos mean", float(cos.mean()), "scalar corr", float(np.corrcoef(rd.ravel(), T[3][:64].ravel())[0, 1]))
log("DONE")
