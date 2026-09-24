"""Faithful feature build for the akhyar2612/0-670 reproduction (raw train CSV + raw VideoMAE npy).
Writes cache/win.npz (stride-25 in-segment windows), cache/tile.npz (every 1-s tile), cache/test.npz.
Limb order inside the cache = akhyar's SENSOR_LOCATIONS = [right_arm, right_leg, left_leg, left_arm]."""
import os
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["OPENBLAS_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"
import sys, time, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from akh_features import extract_acc_features_vect

DATA = r"E:\Claude code\wear\data"; OUT = os.path.join(os.path.dirname(__file__), "cache")
SENSOR_LOCATIONS = ["right_arm", "right_leg", "left_leg", "left_arm"]
SENSOR_TO_ID = {s: i for i, s in enumerate(SENSOR_LOCATIONS)}
CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)", "jogging (butt-kicks)",
               "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
               "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups", "sit-ups (complex)", "burpees",
               "lunges", "lunges (complex)", "bench-dips"]
LABEL_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
INERTIAL_COLS = [f"{loc}_acc_{ax}" for loc in SENSOR_LOCATIONS for ax in "xyz"]
W, VW, VOFF, STRIDE = 50, 15, 8, 25
rng0 = np.random.RandomState(0); PROJ = rng0.randn(1536, 64) / np.sqrt(1536)   # extract_video_features_vect projection


PCA_C = np.load(os.path.join(DATA, "prep", "pca_components.npy")).astype(np.float32)     # (160,768)


def vid_feats(mean768, zframes):
    """akhyar extract_video_features_vect = concat[mean_15frames, std_15frames] @ PROJ (1536->64).
    Raw train frames are not all on disk, so: mean = exact raw mean768 (tiles/test) or average of the two neighbouring
    tiles' means (half-second-offset windows); std = std over frames of the PCA-160 reconstruction (same for train & test)."""
    out = []
    for i in range(0, len(mean768), 2000):
        z = np.asarray(zframes[i:i + 2000], dtype=np.float32)                    # (m,15,160)
        sd = (z @ PCA_C).std(1).astype(np.float64)                              # (m,768)
        out.append(np.concatenate([np.asarray(mean768[i:i + 2000], np.float64), sd], -1) @ PROJ)
    return np.concatenate(out) if out else np.zeros((0, 64))


def win_vid(starts, rows, M768, ZP):
    """video feats for windows starting at inertial sample s (multiple of 25) of a session whose tiles are meta rows `rows`."""
    n = len(rows); t = starts // W; off = (starts % W) > 0
    t1 = np.minimum(t + 1, n - 1)
    mean = M768[rows[t]].astype(np.float64)
    mean[off] = 0.5 * (mean[off] + M768[rows[t1[off]]].astype(np.float64))
    z = ZP[rows[t]].astype(np.float32)
    zo = np.concatenate([ZP[rows[t[off]]][:, 8:15], ZP[rows[t1[off]]][:, 0:8]], 1).astype(np.float32)   # ~frames 23..37
    z[off] = zo
    return vid_feats(mean, z)


def acc_feats(inertial, starts, chunk=4000):
    out = []
    for i in range(0, len(starts), chunk):
        idx = starts[i:i + chunk, None] + np.arange(W)[None]
        w = np.nan_to_num(inertial[idx].astype(np.float64))          # (M,50,4,3)
        out.append(extract_acc_features_vect(w).reshape(len(idx), 4, 100))
    return np.concatenate(out)


def build_windows(labels, stride):
    T = len(labels)
    bounds = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    ss = np.concatenate(([0], bounds)); ee = np.concatenate((bounds, [T]))
    st, tg = [], []
    for s, e in zip(ss, ee):
        lab = labels[s]
        if e - s < W: continue
        first = -(-s // stride) * stride
        if first + W > e: continue
        a = np.arange(first, e - W + 1, stride); st.append(a); tg.append(np.full(len(a), lab, np.int64))
    return np.concatenate(st), np.concatenate(tg)


def main():
    t0 = time.time()
    meta = pd.read_csv(os.path.join(DATA, "prep", "train_meta.csv"))
    imu = np.load(os.path.join(DATA, "prep", "train_imu.npy"), mmap_mode="r")     # ours: [left_arm,left_leg,right_arm,right_leg]
    M768 = np.load(os.path.join(DATA, "prep", "train_vid_mean768.npy"), mmap_mode="r"); ZP = np.load(os.path.join(DATA, "prep", "train_vid_pca.npy"), mmap_mode="r")
    stems = sorted(p[:-4] for p in os.listdir(os.path.join(DATA, "train", "inertial_feat")) if p.endswith(".csv"))
    WF, WV, WY, WR, WS, WST = [], [], [], [], [], []
    TF, TV, TROW, TNAN = [], [], [], []
    for stem in stems:
        df = pd.read_csv(os.path.join(DATA, "train", "inertial_feat", f"{stem}.csv"), low_memory=False, dtype={"label": "string"})
        inertial = df[INERTIAL_COLS].to_numpy(np.float64).reshape(len(df), 4, 3)
        labs = np.array([LABEL_TO_ID.get(x, 0) if isinstance(x, str) else 0 for x in df["label"].to_numpy()], np.int64)
        sbj = int(re.match(r"sbj_(\d+)", stem).group(1))
        starts, tg = build_windows(labs, STRIDE)
        rows = np.where(meta.session.to_numpy() == stem)[0]; n = len(rows)
        WF.append(acc_feats(inertial, starts).astype(np.float32)); WV.append(win_vid(starts, rows, M768, ZP).astype(np.float32))
        WY.append(tg); WR.append(np.full(len(starts), stem)); WS.append(np.full(len(starts), sbj)); WST.append(starts)
        # 1-s tiles aligned with train_meta rows
        assert np.all(meta.t.to_numpy()[rows] == np.arange(n)), stem
        tst = np.arange(n) * W
        assert tst[-1] + W <= len(df), (stem, n, len(df))
        TF.append(acc_feats(inertial, tst).astype(np.float32)); TV.append(win_vid(tst, rows, M768, ZP).astype(np.float32))
        TROW.append(rows)
        # sanity: our train_imu tile == csv block (left_arm = ak 3)
        chk = np.nanmax(np.abs(imu[rows[:200], 0].astype(np.float64) - inertial[:200 * W].reshape(200, W, 4, 3)[:, :, 3]))
        nanl = np.isnan(inertial[:n * W].reshape(n, W, 4, 3)).any(axis=(1, 3))   # (n,4) akhyar order
        TNAN.append(nanl)
        print(f"{stem}: T={len(df)} win={len(starts)} tiles={n} imu_chk={chk:.2e} nan_tiles={nanl.any(1).mean():.3f} {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "win.npz"), F=np.concatenate(WF), V=np.concatenate(WV), y=np.concatenate(WY), rec=np.concatenate(WR),
             sbj=np.concatenate(WS), start=np.concatenate(WST))
    rows = np.concatenate(TROW); order = np.argsort(rows)
    assert np.all(rows[order] == np.arange(len(meta)))
    np.savez(os.path.join(OUT, "tile.npz"), F=np.concatenate(TF)[order], V=np.concatenate(TV)[order], nan=np.concatenate(TNAN)[order])
    # test
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    assert np.all(tm.id.to_numpy() == np.arange(len(tm)))
    ti = np.nan_to_num(np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float64))
    if ti.shape[1] == 3 and ti.shape[2] == 50: ti = ti.transpose(0, 2, 1)
    Ft = extract_acc_features_vect(ti)
    Vt = vid_feats(np.load(os.path.join(DATA, "prep", "test_vid_mean768.npy")), np.load(os.path.join(DATA, "prep", "test_vid_pca.npy"), mmap_mode="r"))
    sid = tm.sensor_location.map(SENSOR_TO_ID).to_numpy(np.int64)
    np.savez(os.path.join(OUT, "test.npz"), F=Ft.astype(np.float32), V=Vt.astype(np.float32), sid=sid, sbj=tm.sbj_id.to_numpy())
    print("windows", sum(len(x) for x in WY), "test", Ft.shape, Vt.shape, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
