"""akhyar2612/0-670 run_hierarchical reproduction: family(8) x variant LGBMs + flat 19-way, 0.7/0.3 blend, null x exp(0.75).
python aka_train.py <folds comma list, e.g. 0,1,2,3,4,full> [--video bug|aligned|none] [--tag name]
  video=bug (default, faithful): train video features are np.tile'd while IMU/labels are np.repeat'ed, exactly as in
  akhyar's feature_data.build_train_features -> train video is misaligned with the labels (row k gets window k mod N).
Outputs: <tag>_cv/oof_f{k}.npy (rows of fold k, (n,4,19) OUR limb order, NaN where limb has NaN IMU), <tag>_full/test.npy."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "4")
import sys, time, argparse
import numpy as np, pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
NC = 19
FAMILIES = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12], [13, 14], [16, 17], [15], [18]]
C2F = np.zeros(NC, np.int64)
for fi, cl in enumerate(FAMILIES):
    for c in cl: C2F[c] = fi
AK_OF_OURS = [3, 2, 0, 1]      # our limb j (left_arm,left_leg,right_arm,right_leg) -> akhyar sensor index
SEED = 42; NJ = 4


def train_lgbm(X, y, n_class, seed, a):
    counts = np.bincount(y, minlength=n_class).astype(float) + 1
    sw = 1.0 / counts[y]; sw = sw / sw.mean()
    m = lgb.LGBMClassifier(objective="multiclass", num_class=n_class, n_estimators=a.n_est, learning_rate=a.lr, num_leaves=63,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=30, n_jobs=NJ,
                           random_state=seed, verbose=-1)
    m.fit(X, y, sample_weight=sw)
    return m


def pad(p, k):
    return p if p.shape[1] >= k else np.concatenate([p, np.zeros((len(p), k - p.shape[1]))], 1)


def fit_predict(Ftr, y, Fte, a, log):
    t0 = time.time(); fam_y = C2F[y]; N = len(Fte)
    fm = train_lgbm(Ftr, fam_y, len(FAMILIES), SEED, a); fam_p = pad(fm.predict_proba(Fte), len(FAMILIES))
    log(f"  family model {time.time()-t0:.0f}s")
    hier = np.zeros((N, NC))
    for fi, classes in enumerate(FAMILIES):
        mask = fam_y == fi
        present = sorted(set(y[mask].tolist()) & set(classes))
        if mask.sum() < 100 or len(present) < 2:
            for c in present: hier[:, c] += fam_p[:, fi] / len(present)
            continue
        lm = {c: j for j, c in enumerate(present)}
        vm = train_lgbm(Ftr[mask], np.array([lm[c] for c in y[mask]]), len(present), SEED + fi + 1, a)
        vp = pad(vm.predict_proba(Fte), len(present))
        for j, c in enumerate(present): hier[:, c] += fam_p[:, fi] * vp[:, j]
        log(f"  variant fam{fi} ({mask.sum()} rows) {time.time()-t0:.0f}s")
    hier = hier / hier.sum(1, keepdims=True)
    cm = train_lgbm(Ftr, y, NC, SEED + 99, a); flat = pad(cm.predict_proba(Fte), NC)
    log(f"  flat model {time.time()-t0:.0f}s")
    probs = hier * (1 - a.flat_w) + flat * a.flat_w; probs = probs / probs.sum(1, keepdims=True)
    probs = probs.copy(); probs[:, 0] *= np.exp(a.null_bias); probs = probs / probs.sum(1, keepdims=True)
    return probs, hier, flat


def build_train(win, keep, a):
    """akhyar subsample_windows(600, seed 42) + per-recording concat order + (N*4) view expansion."""
    df = pd.DataFrame({"i": np.where(keep)[0], "rec": win["rec"][keep], "sbj_id": win["sbj"][keep], "target": win["y"][keep]})
    rng = np.random.RandomState(SEED); df["_r"] = rng.rand(len(df)); df = df.sort_values("_r")
    df["_n"] = df.groupby(["sbj_id", "target"])["_r"].cumcount(); df = df[df["_n"] < a.cap].reset_index(drop=True)
    recs = df["rec"].to_numpy(); order = np.concatenate([np.where(recs == r)[0] for r in pd.unique(recs)])
    idx = df["i"].to_numpy()[order]; M = len(idx)
    X = win["F"][idx].reshape(M * 4, 100); y = np.repeat(win["y"][idx], 4)
    V = win["V"][idx]
    if a.video == "bug": vid = np.tile(V, (4, 1))
    elif a.video == "aligned": vid = np.repeat(V, 4, axis=0)
    else: vid = np.zeros((M * 4, 0), np.float32)
    sensor = np.tile(np.eye(4, dtype=np.float32), (M, 1))
    return np.concatenate([X, vid, sensor], 1), y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folds"); ap.add_argument("--video", default="bug")
    ap.add_argument("--tag", default="aka"); ap.add_argument("--n_est", type=int, default=800); ap.add_argument("--lr", type=float, default=0.07)
    ap.add_argument("--cap", type=int, default=600); ap.add_argument("--flat_w", type=float, default=0.3); ap.add_argument("--null_bias", type=float, default=0.75); ap.add_argument("--nj", type=int, default=4); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--full_dir", default="")
    a = ap.parse_args(); global NJ, SEED; NJ = a.nj; SEED = a.seed
    cvd = os.path.join(HERE, f"{a.tag}_cv"); fd = os.path.join(HERE, a.full_dir or f"{a.tag}_full"); os.makedirs(cvd, exist_ok=True); os.makedirs(fd, exist_ok=True)
    logf = open(os.path.join(HERE, "logs", f"{a.tag}_train_{a.folds.replace(chr(44), chr(95))}.log"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    win = dict(np.load(os.path.join(CACHE, "win.npz")))
    meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
    subjects = np.unique(meta.sbj); perm = np.random.RandomState(0).permutation(subjects); fold = {s: i % 5 for i, s in enumerate(perm)}
    sec_fold = np.array([fold[s] for s in meta.sbj.to_numpy()])
    tile = None
    for f in a.folds.split(","):
        t0 = time.time()
        if f == "full":
            keep = np.ones(len(win["y"]), bool)
            te = np.load(os.path.join(CACHE, "test.npz"))
            Vt = te["V"] if a.video != "none" else np.zeros((len(te["F"]), 0), np.float32)
            Fte = np.concatenate([te["F"], Vt, np.eye(4, dtype=np.float32)[te["sid"]]], 1)
        else:
            k = int(f); keep = np.array([fold[s] != k for s in win["sbj"]])
            if tile is None: tile = dict(np.load(os.path.join(CACHE, "tile.npz")))
            rows = np.where(sec_fold == k)[0]; n = len(rows)
            Vt = tile["V"][rows] if a.video != "none" else np.zeros((n, 0), np.float32)
            Fte = np.concatenate([np.concatenate([tile["F"][rows, s], Vt, np.tile(np.eye(4, dtype=np.float32)[s], (n, 1))], 1) for s in range(4)], 0)
        Ftr, y = build_train(win, keep, a)
        log(f"[{a.tag} fold {f} video={a.video}] train rows {Ftr.shape} classes {np.bincount(y, minlength=NC).tolist()} test rows {Fte.shape}")
        probs, hier, flat = fit_predict(Ftr, y, Fte, a, log)
        if f == "full":
            np.save(os.path.join(fd, "test.npy"), probs.astype(np.float32))
            np.save(os.path.join(fd, "test_hier.npy"), hier.astype(np.float32)); np.save(os.path.join(fd, "test_flat.npy"), flat.astype(np.float32))
        else:
            P = probs.reshape(4, n, NC).transpose(1, 0, 2)                  # (n, akhyar limb, 19)
            out = P[:, AK_OF_OURS].astype(np.float32)                       # our limb order
            nan = tile["nan"][rows][:, AK_OF_OURS]; out[nan] = np.nan
            np.save(os.path.join(cvd, f"oof_f{k}.npy"), out); np.save(os.path.join(cvd, f"rows_f{k}.npy"), rows)
        log(f"[{a.tag} fold {f}] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
