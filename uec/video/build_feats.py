"""Build per-window video features for the UEC Video-MLP/CNN port.

backend pca : 1-s tiles of data/prep (stride 50), frames = PCA-160 reconstructed to 768-d; test from test_vid_pca (same projection).
backend raw : raw per-frame VideoMAE npys, windows at stride 25 (the repo's grid); test from raw test_videomae_data.npy.

Outputs in feats/<backend>/:
  meta.csv       session, sbj, start, tile, y, pur, ra_ok
  a_agg.npy (N,4608) f16, a_sc.npy (N,56) f32   member A: first_mid_last + delta_concat
  b_agg.npy (N,2304) f16, b_sc.npy (N,56) f32   member B: all 15 frames raw
  cnn.npy   (N,3,1536) f16                       Video-CNN input (first_mid_last + delta_concat)
  test_*.npy same, test rows in test-id order
"""
import os, sys, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vport import *

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["pca", "raw"], required=True)
ap.add_argument("--part", choices=["train", "test", "all"], default="all")
ap.add_argument("--sessions", nargs="*", default=None)
args = ap.parse_args()
od = os.path.join(FEAT, args.backend); os.makedirs(od, exist_ok=True)
t0 = time.time()


def feats_for(win15):
    """win15 (B,15,768) f32 -> dict of per-member arrays."""
    a_agg, a_sc = member_features(win15, "a")
    b_agg, b_sc = member_features(win15, "b")
    cnn = build_video_temporal_features(select_frames(win15, "first_mid_last"), "delta_concat")
    return dict(a_agg=a_agg.astype(np.float16), a_sc=a_sc, b_agg=b_agg.astype(np.float16), b_sc=b_sc, cnn=cnn.astype(np.float16))


def run_chunks(get_chunk, n, bs=2048):
    parts = {}
    for s in range(0, n, bs):
        f = feats_for(get_chunk(s, min(n, s + bs)))
        for k, v in f.items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def save(prefix, d):
    for k, v in d.items():
        np.save(os.path.join(od, f"{prefix}{k}.npy"), v)


if args.backend == "pca":
    mean = np.load(os.path.join(PREP, "pca_mean.npy")).astype(np.float32)
    comps = np.load(os.path.join(PREP, "pca_components.npy")).astype(np.float32)
    recon = lambda z: (z.astype(np.float32) @ comps + mean)
    if args.part in ("train", "all"):
        meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
        imu = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r")      # (N,4,50,3) limbs la,ll,ra,rl
        ra_ok = np.isfinite(np.asarray(imu[:, 2], np.float32)).all(axis=(1, 2))
        m = pd.DataFrame({"session": meta.session, "sbj": meta.sbj, "start": meta.t * 50, "tile": meta.t,
                          "y": meta.y, "pur": meta.pur, "ra_ok": ra_ok.astype(np.int8)})
        m.to_csv(os.path.join(od, "meta.csv"), index=False)
        Z = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
        d = run_chunks(lambda s, e: recon(np.asarray(Z[s:e])), len(Z))
        save("", d)
        print("train", {k: v.shape for k, v in d.items()}, f"{time.time()-t0:.0f}s", flush=True)
    if args.part in ("test", "all"):
        Z = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r")
        d = run_chunks(lambda s, e: recon(np.asarray(Z[s:e])), len(Z))
        tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
        assert (tm.id.to_numpy() == np.arange(len(tm))).all(), "test meta not in id order"
        save("test_", d)
        print("test", {k: v.shape for k, v in d.items()}, f"{time.time()-t0:.0f}s", flush=True)

else:  # raw
    vdir = os.path.join(DATA, "train", "videomae_feat"); cdir = os.path.join(DATA, "train", "inertial_feat")
    sd = os.path.join(od, "sessions"); os.makedirs(sd, exist_ok=True)
    if args.part in ("train", "all"):
        sessions = args.sessions or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(cdir, "sbj_*.csv")))
        for sess in sessions:
            if os.path.exists(os.path.join(sd, f"{sess}_meta.csv")):
                print("skip (done)", sess, flush=True); continue
            vp = os.path.join(vdir, f"{sess}.npy")
            try:
                V = np.load(vp, mmap_mode="r")
            except Exception as ex:
                print("video not ready", sess, ex, flush=True); continue
            df = pd.read_csv(os.path.join(cdir, f"{sess}.csv"), low_memory=False, keep_default_na=False,
                             usecols=["sbj_id", "right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z", "label"])
            lab = encode_labels(df["label"])
            ra = df[["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            n = len(df)
            if V.shape[0] < 0.55 * n * VIDEO_HZ / IMU_HZ:
                print("video too short / incomplete", sess, V.shape, n, flush=True); continue
            starts = np.arange(0, n - WIN + 1, 25)
            s0 = np.array([center_crop_indices(s)[0] for s in starts])
            keep = (s0 >= 0) & (s0 + VWIN <= V.shape[0])
            starts, s0 = starts[keep], s0[keep]
            y, pur = window_labels(lab, starts)
            fin = np.isfinite(ra)
            ra_ok = np.array([fin[s:s + WIN].all() for s in starts])
            idx = s0[:, None] + np.arange(VWIN)[None, :]
            d = run_chunks(lambda a, b: np.nan_to_num(np.asarray(V[idx[a:b].reshape(-1)], np.float32)).reshape(b - a, VWIN, -1), len(starts))
            for k, v in d.items():
                np.save(os.path.join(sd, f"{sess}_{k}.npy"), v)
            m = pd.DataFrame({"session": sess, "sbj": int(df.sbj_id.iloc[0]), "start": starts,
                              "tile": np.where(starts % 50 == 0, starts // 50, -1), "y": y, "pur": pur, "ra_ok": ra_ok.astype(np.int8)})
            m.to_csv(os.path.join(sd, f"{sess}_meta.csv"), index=False)
            print(sess, len(starts), f"frames={V.shape[0]} samples={n} {time.time()-t0:.0f}s", flush=True)
    if args.part in ("test", "all"):
        X = np.load(os.path.join(DATA, "test", "test_videomae_data.npy"), mmap_mode="r")
        tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
        assert (tm.id.to_numpy() == np.arange(len(tm))).all(), "test meta not in id order"
        d = run_chunks(lambda s, e: np.nan_to_num(np.asarray(X[s:e], np.float32).transpose(0, 2, 1)), len(X), bs=1024)
        save("test_", d)
        print("test", {k: v.shape for k, v in d.items()}, f"{time.time()-t0:.0f}s", flush=True)
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
