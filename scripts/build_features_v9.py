#!/usr/bin/env python3
"""WEAR v9 feature builder v2 — limb-matched single-sensor formulation.

Row layout: limb-row index = global_window_idx * 4 + limb_idx.
Step 'train': per-limb IMU features + labels + window video features.
Step 'test' : per-window features from its known limb.
Step 'knn'  : video PCA + fold-safe kNN label-histogram features (train OOF + test).
"""
import os, sys, glob
import numpy as np
import pandas as pd

DATA = "/home/z/my-project/data"
PROC = f"{DATA}/proc"
os.makedirs(PROC, exist_ok=True)

CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)",
    "jogging (sidesteps)", "jogging (butt-kicks)", "stretching (triceps)",
    "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
    "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
    "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LAB2ID = {c: i for i, c in enumerate(CLASS_NAMES)}
LIMBS = ["right_arm", "right_leg", "left_leg", "left_arm"]
WINDOW, STRIDE = 50, 25

def axis_stats(X):
    mu = X.mean(1); sd = X.std(1)
    d = X - mu[:, None]
    m2 = (d ** 2).mean(1); m3 = (d ** 3).mean(1); m4 = (d ** 4).mean(1)
    skew = m3 / (m2 ** 1.5 + 1e-9)
    kurt = m4 / (m2 ** 2 + 1e-9) - 3.0
    med = np.median(X, 1)
    q1 = np.percentile(X, 25, 1); q3 = np.percentile(X, 75, 1)
    mad = np.median(np.abs(X - med[:, None]), 1)
    rms = np.sqrt((X ** 2).mean(1))
    mn = X.min(1); mx = X.max(1)
    T = X.shape[1]
    t = np.arange(T, dtype=np.float32); tc = t - t.mean()
    slope = (d * tc[None, :]).sum(1) / (tc ** 2).sum()
    zcr = (np.diff(np.sign(d), axis=1) != 0).mean(1)
    dx = np.diff(X, 1)
    jstd = dx.std(1); jmax = np.abs(dx).max(1); jrms = np.sqrt((dx ** 2).mean(1))
    return [mu, sd, mn, mx, med, q1, q3, mad, rms, skew, kurt, slope, zcr, jstd, jmax, jrms]

def spec_feats(X, fs=50.0, bands=((0, 1), (1, 3), (3, 6), (6, 12), (12, 25))):
    n, T = X.shape
    Ff = np.fft.rfft(X - X.mean(1, keepdims=True), axis=1)
    P = (np.abs(Ff) ** 2)[:, :T // 2]
    freqs = np.fft.rfftfreq(T, 1 / fs)[:T // 2]
    tot = P.sum(1) + 1e-9
    dom = freqs[P.argmax(1)]
    out = [dom, np.log1p(tot)]
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out.append(np.log1p(P[:, m].sum(1)))
    Ps = P / tot[:, None]
    sent = -(np.where(Ps > 0, Ps * np.log(Ps + 1e-12), 0)).sum(1)
    out.append(sent)
    return out

def limb_features(S):
    outs = []
    for ax in range(3):
        outs += axis_stats(S[:, :, ax])
        outs += spec_feats(S[:, :, ax])
    k = 9; c = np.ones(k, dtype=np.float32) / k
    for ax in range(3):
        pad = np.apply_along_axis(lambda r: np.convolve(r, c, mode="same"), 1, S[:, :, ax])
        outs += [pad.mean(1), (S[:, :, ax] - pad).std(1)]
    mag = np.linalg.norm(S, axis=2)
    outs += axis_stats(mag)
    outs += spec_feats(mag)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        xa = S[:, :, a] - S[:, :, a].mean(1, keepdims=True)
        xb = S[:, :, b] - S[:, :, b].mean(1, keepdims=True)
        corr = (xa * xb).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1) + 1e-9)
        outs.append(corr)
    return np.stack(outs, 1).astype(np.float32)

def step_train():
    sbjs = sorted([os.path.basename(p)[:-4] for p in glob.glob(f"{DATA}/train/inertial_feat/*.csv")])
    per_sbj_limb = []   # per_sbj_limb[si][li] -> (n_win_si, F)
    labels, vms, vss = [], [], []
    total = 0
    for sbj in sbjs:
        df = pd.read_csv(f"{DATA}/train/inertial_feat/{sbj}.csv")
        cols = [f"{l}_acc_{a}" for l in LIMBS for a in "xyz"]
        V = df[cols].to_numpy(dtype=np.float32)
        y = df["label"].map(LAB2ID).to_numpy(dtype=np.int64)
        n_win = len(V) // STRIDE - 1
        W = np.lib.stride_tricks.sliding_window_view(V, (WINDOW, 12))[::STRIDE, 0][:n_win].copy()
        W = W.reshape(n_win, WINDOW, 4, 3)
        fl = [limb_features(W[:, :, li, :]) for li in range(4)]
        per_sbj_limb.append(fl)
        labels.append(y[::STRIDE][:n_win].astype(np.int8))
        # video window features: chunks (k, k+1) weights (7, 8)
        vm = np.load(f"{DATA}/train/videomae_pooled/{sbj}_mean.npy").astype(np.float32)
        vs = np.load(f"{DATA}/train/videomae_pooled/{sbj}_std.npy").astype(np.float32)
        n_vid = min(n_win, len(vm) - 1)
        n = min(n_win, n_vid)
        vw_m = (7 * vm[:n] + 8 * vm[1:n + 1]) / 15.0
        var = (7 * (vs[:n] ** 2 + vm[:n] ** 2) + 8 * (vs[1:n + 1] ** 2 + vm[1:n + 1] ** 2)) / 15.0
        vw_s = np.sqrt(np.maximum(var - vw_m ** 2, 0))
        vms.append(vw_m); vss.append(vw_s)
        total += n
        print(f"{sbj}: wins={n}", flush=True)
    y = np.concatenate(labels); vm = np.concatenate(vms); vsd = np.concatenate(vss)
    n_win = len(y)
    F = np.zeros((n_win * 4, per_sbj_limb[0][0].shape[1]), np.float32)
    wg = 0
    for si, fl in enumerate(per_sbj_limb):
        n = fl[0].shape[0]
        for li in range(4):
            F[(wg) * 4 + li: (wg + n) * 4 + li: 4] = fl[li][:n]
        wg += n
    np.save(f"{PROC}/train_limbfX.npy", F)
    np.save(f"{PROC}/train_y.npy", y)
    np.save(f"{PROC}/train_vmean.npy", vm.astype(np.float16))
    np.save(f"{PROC}/train_vstd.npy", vsd.astype(np.float16))
    with open(f"{PROC}/sbjs.txt", "w") as f:
        f.write("\n".join(sbjs))
    # subject id per window and per limb-row
    counts = [fl[0].shape[0] for fl in per_sbj_limb]
    sb = np.concatenate([np.full(c, i, np.int8) for i, c in enumerate(counts)])
    np.save(f"{PROC}/train_sbj.npy", sb)
    print("TRAIN:", F.shape, "windows:", n_win, "nfeat:", F.shape[1], flush=True)

def step_test():
    x = np.load(f"{DATA}/test/test_inertial_data.npy", mmap_mode="r")
    tm = pd.read_csv(f"{DATA}/test/test_meta_data.csv")
    F = None; limbcol = np.zeros(len(tm), np.int8)
    for li, l in enumerate(LIMBS):
        mask = (tm.sensor_location.to_numpy() == l)
        S = np.asarray(x[mask], dtype=np.float32)
        fl = limb_features(S)
        if F is None:
            F = np.zeros((len(tm), fl.shape[1]), np.float32)
        F[mask] = fl
        limbcol[mask] = li
        print(f"test {l}: n={mask.sum()}", flush=True)
    np.save(f"{PROC}/test_limbfX.npy", F)
    np.save(f"{PROC}/test_limbcol.npy", limbcol)
    np.save(f"{PROC}/test_vmean.npy", np.load(f"{DATA}/test/vid_mean.npy"))
    np.save(f"{PROC}/test_vstd.npy", np.load(f"{DATA}/test/vid_std.npy"))
    print("TEST:", F.shape, flush=True)

def step_knn():
    """Video PCA + fold-safe kNN label-histogram features."""
    from sklearn.decomposition import TruncatedSVD
    if os.path.exists(f"{PROC}/video_tr_pca.npy"):
        Ztr = np.load(f"{PROC}/video_tr_pca.npy"); Zte = np.load(f"{PROC}/video_te_pca.npy")
    else:
        vm_tr = np.load(f"{PROC}/train_vmean.npy").astype(np.float32)
        vs_tr = np.load(f"{PROC}/train_vstd.npy").astype(np.float32)
        vm_te = np.load(f"{PROC}/test_vmean.npy").astype(np.float32)
        vs_te = np.load(f"{PROC}/test_vstd.npy").astype(np.float32)
        # standardized PCA space (fit on train windows)
        Xtr = np.concatenate([vm_tr, vs_tr], 1)
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        svd = TruncatedSVD(n_components=48, random_state=0)
        Ztr = svd.fit_transform(Xtr_n).astype(np.float32)
        Zte = svd.transform((np.concatenate([vm_te, vs_te], 1) - mu) / sd).astype(np.float32)
        np.save(f"{PROC}/video_tr_pca.npy", Ztr); np.save(f"{PROC}/video_te_pca.npy", Zte)
        print("PCA done, explained var:", svd.explained_variance_ratio_.sum().round(3), flush=True)
    y = np.load(f"{PROC}/train_y.npy").astype(np.int64)
    sb = np.load(f"{PROC}/train_sbj.npy").astype(np.int64)
    # L2 normalize for cosine
    Ztr_n = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
    Zte_n = Zte / (np.linalg.norm(Zte, axis=1, keepdims=True) + 1e-9)
    K = 15; NCLASS = 19

    def knn_feats(Zq, Zdb, y_db, chunk=512):
        n = Zq.shape[0]
        hist = np.zeros((n, NCLASS), np.float32)
        whist = np.zeros((n, NCLASS), np.float32)
        dmean5 = np.zeros(n, np.float32); dmean15 = np.zeros(n, np.float32)
        for st in range(0, n, chunk):
            en = min(n, st + chunk)
            sim = Zq[st:en] @ Zdb.T  # cosine
            idx = np.argpartition(-sim, K, axis=1)[:, :K]
            rows = np.arange(en - st)[:, None]
            s = sim[rows, idx]
            for j in range(K):
                hist[st:en] += np.eye(NCLASS, dtype=np.float32)[y_db[idx[:, j]]]
                whist[st:en] += np.eye(NCLASS, dtype=np.float32)[y_db[idx[:, j]]] * (s[:, j:j + 1] + 1.0)
            dmean5[st:en] = 1 - s[:, :5].mean(1)
            dmean15[st:en] = 1 - s.mean(1)
            hist[st:en] /= K; whist[st:en] /= (whist[st:en].sum(1, keepdims=True) + 1e-9)
        return np.concatenate([hist, whist, dmean5[:, None], dmean15[:, None]], 1).astype(np.float32)

    # fold-safe OOF: for each subject, neighbors exclude that subject
    sbjs = np.unique(sb)
    oof_knn = np.zeros((len(y), 2 * NCLASS + 2), np.float32)
    for s in sbjs:
        q = sb == s
        db = ~q
        oof_knn[q] = knn_feats(Ztr_n[q], Ztr_n[db], y[db])
        print(f"knn fold-out sbj {s}: n={q.sum()}", flush=True)
    np.save(f"{PROC}/knn_tr.npy", oof_knn)
    knn_te = knn_feats(Zte_n, Ztr_n, y)
    np.save(f"{PROC}/knn_te.npy", knn_te)
    print("KNN done:", oof_knn.shape, knn_te.shape, flush=True)

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("all", "train"): step_train()
    if step in ("all", "test"): step_test()
    if step in ("all", "knn"): step_knn()
