"""Build the UEC Inertial-LightGBM feature tables (exact repo features via vfeat, verified vs reference).

Per train session (all 24 CSVs):
  windows of 50 samples, stride 25 (repo: --window-size 50 --stride 25), label = majority, purity = majority share.
  A feature row is written for each (start, sensor) whose sensor window is finite AND purity >= 0.8, and
  (session is not a *_2 session OR start % 50 == 0)  [the _2 tiles are only used for our standard-protocol eval].
  meta: start, sensor (ra/rl/ll/la), y, pur, all4 (all 4 sensors finite at that start), repo (row the repo trains on:
        purity>=0.8, all 4 sensors finite, session not *_2).
Test: 12234 windows, one sensor each (repo build_test_table, normalization none).
Outputs: feats/<session>.npz, feats/test.npz, feats/names.json
"""
import os, sys, json, time, glob
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, r"E:\Claude code\wear\uec\gbdt")
OUT = r"E:\Claude code\wear\uec\gbdt\feats"
DATA = r"E:\Claude code\wear\data"
SENSORS = ["ra", "rl", "ll", "la"]          # repo SENSOR_COLS order
WS, STRIDE, PUR = 50, 25, 0.8
CHUNK = 4096


def build_session(path):
    import pandas as pd
    from uecstub import tig
    import vfeat
    t0 = time.time()
    sess = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT, f"{sess}.npz")
    if os.path.exists(out):
        return sess, "skip", 0
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)
    sbj = int(df["sbj_id"].iloc[0])
    lab = np.array([-1 if v is None else v for v in (tig.encode_label(v) for v in df["label"])], dtype=np.int64)
    arrs = {k: df[c].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) for k, c in tig.SENSOR_COLS.items()}
    n = len(df)
    starts = np.arange(0, n - WS + 1, STRIDE)
    win_idx = starts[:, None] + np.arange(WS)[None, :]
    L = lab[win_idx]                                            # (S,50)
    has_none = (L < 0).any(1)
    counts = np.stack([(L == c).sum(1) for c in range(tig.N_CLASSES)], 1)
    y = counts.argmax(1)
    pur = counts.max(1) / float(WS)
    fin = {k: np.isfinite(arrs[k][win_idx]).all(axis=(1, 2)) for k in SENSORS}
    all4 = np.logical_and.reduce([fin[k] for k in SENSORS])
    is2 = tig.should_exclude_file_id_suffix_2(sess)
    keep_start = (~has_none) & (pur >= PUR) & ((not is2) | (starts % 50 == 0))
    rows_s, rows_k = [], []
    for si in np.where(keep_start)[0]:
        for k in SENSORS:
            if fin[k][si]:
                rows_s.append(si)
                rows_k.append(k)
    rows_s = np.array(rows_s, dtype=np.int64)
    rows_k = np.array(rows_k)
    Xs = []
    names = None
    for c0 in range(0, len(rows_s), CHUNK):
        ss = rows_s[c0:c0 + CHUNK]
        kk = rows_k[c0:c0 + CHUNK]
        W = np.empty((len(ss), WS, 3), np.float32)
        for k in SENSORS:
            m = kk == k
            if m.any():
                W[m] = arrs[k][win_idx[ss[m]]]
        names, X = vfeat.make_feature_matrix(W, kk, smoothing_window=5, sensor_embedding_mode="sensor")
        Xs.append(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    X = np.concatenate(Xs)
    np.savez(out, X=X, start=starts[rows_s].astype(np.int32), sensor=rows_k, y=y[rows_s].astype(np.int8),
             pur=pur[rows_s].astype(np.float32), all4=all4[rows_s], repo=(all4[rows_s] & (not is2)),
             sbj=np.full(len(rows_s), sbj, np.int16), n_samples=np.int64(n))
    with open(os.path.join(OUT, "names.json"), "w") as f:
        json.dump(names, f)
    return sess, f"rows={len(rows_s)} starts={len(starts)} kept_starts={int(keep_start.sum())} " \
                 f"none={int(has_none.sum())} all4_fail={int((~all4).sum())}", time.time() - t0


def build_test():
    import pandas as pd
    from uecstub import tig
    import vfeat
    out = os.path.join(OUT, "test.npz")
    if os.path.exists(out):
        return
    Xt = np.load(os.path.join(DATA, "test", "test_inertial_data.npy"))
    meta = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    assert (meta["id"].to_numpy() == np.arange(len(meta))).all(), "test meta not in id order"
    keys = np.array([tig.normalize_sensor_location(s) for s in meta["sensor_location"]])
    W = np.asarray(Xt, dtype=np.float32)
    if W.shape[1:] == (3, 50):
        W = W.transpose(0, 2, 1)
    names, X = vfeat.make_feature_matrix(W, keys, smoothing_window=5, sensor_embedding_mode="sensor")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.savez(out, X=X, id=meta["id"].to_numpy(), sensor=keys, sbj=meta["sbj_id"].to_numpy())
    print("test", X.shape, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_test()
    paths = sorted(glob.glob(os.path.join(DATA, "train", "inertial_feat", "sbj_*.csv")),
                   key=lambda p: -os.path.getsize(p))
    with Pool(int(os.environ.get("NPROC", "3"))) as pool:
        for sess, msg, dt in pool.imap_unordered(build_session, paths):
            print(f"{sess}: {msg} {dt:.0f}s", flush=True)
    print("done", flush=True)
