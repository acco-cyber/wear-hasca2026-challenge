"""abhinavm2811 'Fast Vectorized Pipeline' feature rebuild from local prep data.
Per (second, limb): 52 inertial stats (exact notebook code) + 4-d location one-hot (notebook index order
right_arm0,right_leg1,left_leg2,left_arm3); per second: video mean(768)+std(768) over the 15 frames.
Local deviations (unavoidable): train frames = central 15 (notebook: 15 linspace frames over the whole second);
video mean = raw 768 mean over the central 15 frames (train & test prep files, float16);
video std = std over frames of the PCA-160 reconstruction (same for train and test; raw train frames not local);
train IMU is float16 -> test IMU is rounded through float16 too; label = y_c (label at the window centre sample).
Writes cache/train.npz (Fi (N,4,56), V (N,1536), y, sbj, nan) and cache/test.npz (X (12234,1592))."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "3")
import time
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
P = r"E:\Claude code\wear\data\prep"; T = r"E:\Claude code\wear\data\test"
LOCATION_NAMES = ["right_arm", "right_leg", "left_leg", "left_arm"]
OURS_TO_NB = [3, 2, 0, 1]   # our limb j (left_arm,left_leg,right_arm,right_leg) -> notebook location index


def location_to_index(loc_str):
    s = str(loc_str).strip().lower().replace("-", "_").replace(" ", "_")
    for idx, name in enumerate(LOCATION_NAMES):
        if name in s or s in name:
            return idx
    raise ValueError(loc_str)


def extract_inertial_features_batch(batch):
    """exact copy of the notebook function. batch (N,T,3) -> (N,52)"""
    feats = []
    eps = 1e-8
    for c in range(3):
        sig = batch[:, :, c]
        feats.append(sig.mean(axis=1))
        feats.append(sig.std(axis=1))
        feats.append(sig.min(axis=1))
        feats.append(sig.max(axis=1))
        feats.append(np.median(sig, axis=1))
        feats.append(np.nan_to_num(skew(sig, axis=1), nan=0.0))
        feats.append(np.nan_to_num(kurtosis(sig, axis=1), nan=0.0))
        feats.append(np.sqrt(np.mean(sig ** 2, axis=1)))
        feats.append(np.sum(np.abs(np.diff(sig, axis=1)), axis=1))
        feats.append(np.percentile(sig, 25, axis=1))
        feats.append(np.percentile(sig, 75, axis=1))
        fft_mag = np.abs(rfft(sig, axis=1))
        n_bands = 4
        edges = np.linspace(0, fft_mag.shape[1], n_bands + 1).astype(int)
        for i in range(n_bands):
            feats.append(fft_mag[:, edges[i]:edges[i + 1]].sum(axis=1))
    ax, ay, az = batch[:, :, 0], batch[:, :, 1], batch[:, :, 2]
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    feats.append(mag.mean(axis=1))
    feats.append(mag.std(axis=1))
    feats.append(mag.max(axis=1))
    feats.append(np.sqrt(np.mean(mag ** 2, axis=1)))

    def batch_corr(a, b):
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        cov = (ac * bc).mean(axis=1)
        denom = a.std(axis=1) * b.std(axis=1)
        return np.where(denom > eps, cov / np.where(denom > eps, denom, 1.0), 0.0)

    feats.append(batch_corr(ax, ay))
    feats.append(batch_corr(ax, az))
    feats.append(batch_corr(ay, az))
    return np.stack(feats, axis=1).astype(np.float32)


def vid_std(Z, C, chunk=3000):
    """Z (n,15,160) PCA coords -> std over frames of the 768-d reconstruction (mean offset cancels)."""
    out = np.zeros((len(Z), C.shape[1]), np.float32)
    for a in range(0, len(Z), chunk):
        z = np.asarray(Z[a:a + chunk], dtype=np.float32)
        z = z - z.mean(axis=1, keepdims=True)
        R = z @ C                                   # (n,15,768)
        out[a:a + chunk] = np.sqrt(np.mean(R ** 2, axis=1))
    return out


def main():
    t0 = time.time(); os.makedirs(CACHE, exist_ok=True)
    C = np.load(os.path.join(P, "pca_components.npy")).astype(np.float32)   # (160,768)
    meta = pd.read_csv(os.path.join(P, "train_meta.csv"))
    imu = np.load(os.path.join(P, "train_imu.npy")).astype(np.float32)     # (N,4,50,3)
    N = len(imu)
    Fi = np.zeros((N, 4, 56), np.float32)
    with np.errstate(all="ignore"):
        for j in range(4):
            f = extract_inertial_features_batch(imu[:, j])
            oh = np.zeros((N, 4), np.float32); oh[:, OURS_TO_NB[j]] = 1.0
            Fi[:, j] = np.concatenate([f, oh], 1)
            print(f"train limb {j} feats {time.time()-t0:.0f}s", flush=True)
    nan = np.isnan(imu).any(axis=(2, 3))
    del imu
    vm = np.load(os.path.join(P, "train_vid_mean768.npy")).astype(np.float32)
    vs = vid_std(np.load(os.path.join(P, "train_vid_pca.npy"), mmap_mode="r"), C)
    V = np.concatenate([vm, vs], 1); del vm, vs
    print(f"train video {V.shape} {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(CACHE, "train.npz"), Fi=Fi, V=V, y=meta.y_c.to_numpy(np.int64), y_maj=meta.y.to_numpy(np.int64),
             sbj=meta.sbj.to_numpy(np.int64), nan=nan)
    del Fi, V

    tm = pd.read_csv(os.path.join(T, "test_meta_data.csv"))
    assert (tm.id.to_numpy() == np.arange(len(tm))).all()
    ti = np.load(os.path.join(T, "test_inertial_data.npy")).astype(np.float16).astype(np.float32)   # (12234,50,3)
    loc = np.array([location_to_index(s) for s in tm.sensor_location])
    with np.errstate(all="ignore"):
        fi = extract_inertial_features_batch(ti)
    oh = np.zeros((len(loc), 4), np.float32); oh[np.arange(len(loc)), loc] = 1.0
    vm = np.load(os.path.join(P, "test_vid_mean768.npy")).astype(np.float32)
    vs = vid_std(np.load(os.path.join(P, "test_vid_pca.npy"), mmap_mode="r"), C)
    X = np.concatenate([fi, oh, vm, vs], 1).astype(np.float32)
    # sanity: raw test std (true central-15 std) vs PCA-reconstructed std
    raw = np.load(os.path.join(T, "test_videomae_data.npy"), mmap_mode="r")
    rs = np.asarray(raw[:2000], dtype=np.float32).std(axis=2)
    print("test std raw vs pca-rec (first 2000): mean", float(rs.mean()), float(vs[:2000].mean()),
          "corr", float(np.corrcoef(rs.ravel(), vs[:2000].ravel())[0, 1]), flush=True)
    rm = np.asarray(raw[:2000], dtype=np.float32).mean(axis=2)
    print("test mean prep vs raw max abs diff", float(np.abs(rm - vm[:2000]).max()), flush=True)
    np.savez(os.path.join(CACHE, "test.npz"), X=X, loc=loc)
    print(f"test {X.shape} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
