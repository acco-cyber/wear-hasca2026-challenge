"""Subject-disjoint CV LightGBM per-window classifier (limb-matched rows).
Inputs (from the Kaggle prep kernel, in data/prep/): train_meta.csv, train_imu.npy, train_vid_pca.npy, test_vid_pca.npy
Local: data/test/test_inertial_data.npy, data/test/test_meta_data.csv
Outputs in work/<tag>/: oof.npy (N,4,19) float32 (NaN where limb missing), test.npy (12234,19), feats cache, cv.json
Usage: python train_lgbm.py [tag] [--folds 5] [--rounds 1500]
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from imu_feats import limb_features

DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep"); WORK = r"E:\Claude code\wear\work"
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
NC = 19

def video_feats(Z):
    """Z: (n,15,K) -> (n, 3K): mean, std, (last3-first3) over frames."""
    Z = np.asarray(Z, np.float32)
    return np.concatenate([Z.mean(1), Z.std(1), (Z[:, -3:].mean(1) - Z[:, :3].mean(1))[:, :48]], 1)

def build_train(cache):
    if os.path.exists(cache):
        d = np.load(cache); return d["X"], d["y"], d["sec"], d["limb"], d["sbj"], d["pur"]
    meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    imu = np.load(os.path.join(PREP, "train_imu.npy"))            # (N,4,50,3) f16
    vid = np.load(os.path.join(PREP, "train_vid_pca.npy"))        # (N,15,K) f16
    N = len(meta); print("seconds", N, "K", vid.shape[-1], flush=True)
    VF = video_feats(vid); del vid
    Xs, ys, secs, limbs = [], [], [], []
    for li in range(4):
        F = limb_features(imu[:, li])
        oh = np.zeros((N, 4), np.float32); oh[:, li] = 1
        Xs.append(np.concatenate([F, oh, VF], 1)); ys.append(meta.y.to_numpy()); secs.append(np.arange(N)); limbs.append(np.full(N, li))
        print("limb", li, "feats", F.shape, flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.int64); sec = np.concatenate(secs); limb = np.concatenate(limbs)
    sbj = meta.sbj.to_numpy()[sec]; pur = meta.pur.to_numpy()[sec]
    np.savez(cache, X=X, y=y, sec=sec, limb=limb, sbj=sbj, pur=pur)
    return X, y, sec, limb, sbj, pur

def build_test():
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float32)
    vid = np.load(os.path.join(PREP, "test_vid_pca.npy"))
    F = limb_features(xi)
    oh = np.zeros((len(tm), 4), np.float32)
    for li, l in enumerate(LIMBS): oh[tm.sensor_location.to_numpy() == l, li] = 1
    return np.concatenate([F, oh, video_feats(vid)], 1), tm

def subject_normalise(X, groups, ref_groups=None):
    """Robust per-group standardisation: (x - median_g) / (IQR_g + eps), computed over all rows of the group.
    Returns a copy. NaN-safe."""
    X = X.copy()
    for g in np.unique(groups):
        m = groups == g
        med = np.nanmedian(X[m], 0); q1, q3 = np.nanpercentile(X[m], [25, 75], axis=0)
        X[m] = (X[m] - med) / (q3 - q1 + 1e-3)
    return X

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag", nargs="?", default="lgbm_v1")
    ap.add_argument("--subj_norm", action="store_true", help="per-(session,limb) robust normalisation of all features; test per (subject,limb)")
    ap.add_argument("--keep_raw_video", action="store_true", help="with --subj_norm also keep the un-normalised pooled video block")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--null_x", type=float, default=3.0, help="null rows kept = null_x * median activity count")
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min_pur", type=float, default=0.8); ap.add_argument("--limbs_per_sec", type=int, default=2)
    a = ap.parse_args()
    out = os.path.join(WORK, a.tag); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    X, y, sec, limb, sbj, pur = build_train(os.path.join(PREP, "train_rows_cache.npz"))
    Xte, tm = build_test()
    if a.subj_norm:
        meta0 = pd.read_csv(os.path.join(PREP, "train_meta.csv")); sess = meta0.session.to_numpy()[sec]
        grp_tr = np.array([f"{s}|{l}" for s, l in zip(sess, limb)])
        te_limb = np.array([LIMBS.index(l) for l in tm.sensor_location]); grp_te = np.array([f"{s}|{l}" for s, l in zip(tm.sbj_id, te_limb)])
        nf = 110 + 4; Xn = subject_normalise(X[:, :nf], grp_tr); Xten = subject_normalise(Xte[:, :nf], grp_te)
        # video block normalised per session / per test subject (limb-independent)
        Vn = subject_normalise(X[:, nf:], sess); Vten = subject_normalise(Xte[:, nf:], tm.sbj_id.to_numpy())
        if a.keep_raw_video:
            X = np.concatenate([Xn, Vn, X[:, nf:]], 1); Xte = np.concatenate([Xten, Vten, Xte[:, nf:]], 1)
        else:
            X = np.concatenate([Xn, Vn], 1); Xte = np.concatenate([Xten, Vten], 1)
        X[:, 110:114] = np.eye(4, dtype=np.float32)[limb]; Xte[:, 110:114] = np.eye(4, dtype=np.float32)[te_limb]   # restore one-hot
        print("subject-normalised features", X.shape, flush=True)
    ok = ~np.isnan(X[:, 0]) & (pur >= a.min_pur)
    # training-row thinning: keep `a.limbs_per_sec` random limbs per second for TRAINING (OOF still predicted on all limbs)
    rng0 = np.random.RandomState(1); keep_tr = np.zeros(len(X), bool)
    N_sec0 = sec.max() + 1
    for li_keep in range(a.limbs_per_sec):
        pick = rng0.randint(0, 4, N_sec0)
        keep_tr |= (limb == pick[sec])
    print("rows", len(X), "usable", ok.sum(), "train-thinned", (ok & keep_tr).sum(), "feats", X.shape[1], f"{time.time()-t0:.0f}s", flush=True)
    N_sec = sec.max() + 1
    subjects = np.unique(sbj); rng = np.random.RandomState(0); perm = rng.permutation(subjects)
    fold_of = {s: i % a.folds for i, s in enumerate(perm)}
    fold = np.array([fold_of[s] for s in sbj])
    oof = np.full((N_sec, 4, NC), np.nan, np.float32); test = np.zeros((len(Xte), NC), np.float32)
    params = dict(objective="multiclass", num_class=NC, metric="multi_logloss", learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=60, feature_fraction=0.3, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  max_bin=63, num_threads=12, verbose=-1, seed=0)
    def flush_log(period=25):
        def cb(env):
            if env.iteration % period == 0 and env.evaluation_result_list:
                print(f"    it {env.iteration} {env.evaluation_result_list[0][1]} {env.evaluation_result_list[0][2]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return cb
    cv = {}
    for k in range(a.folds):
        tr = ok & keep_tr & (fold != k); va = ok & (fold == k)
        # null undersampling on the training side
        cnt = np.bincount(y[tr], minlength=NC); med = np.median(cnt[1:]); keep_null = int(a.null_x * med)
        nidx = np.where(tr & (y == 0))[0]
        drop = rng.choice(nidx, max(0, len(nidx) - keep_null), replace=False) if len(nidx) > keep_null else []
        tr2 = tr.copy(); tr2[drop] = False
        cnt2 = np.bincount(y[tr2], minlength=NC).astype(np.float64)
        w_cls = (cnt2.mean() / np.maximum(cnt2, 1)) ** 0.5
        w = w_cls[y[tr2]]
        dtr = lgb.Dataset(X[tr2], y[tr2], weight=w); dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), flush_log(25)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        oof[sec[va], limb[va]] = p
        f1 = f1_score(y[va], p.argmax(1), average="macro")
        test += m.predict(Xte, num_iteration=m.best_iteration) / a.folds
        cv[k] = dict(best_iter=m.best_iteration, f1_rows=float(f1), n_tr=int(tr2.sum()), n_va=int(va.sum()), subjects=[int(s) for s in subjects if fold_of[s] == k])
        print(f"fold {k}: iter {m.best_iteration} row-F1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        m.save_model(os.path.join(out, f"model_f{k}.txt"))
    # window-level OOF F1 with one random limb per second (test-like) and with 4-limb mean
    meta = pd.read_csv(os.path.join(PREP, "train_meta.csv")); yy = meta.y.to_numpy()
    valid = ~np.isnan(oof[:, :, 0])
    rl = np.array([rng.choice(np.where(v)[0]) if v.any() else -1 for v in valid])
    m1 = rl >= 0
    p1 = oof[np.arange(N_sec)[m1], rl[m1]]
    f1_single = f1_score(yy[m1], p1.argmax(1), average="macro")
    pm = np.nanmean(np.log(np.clip(oof, 1e-6, 1)), 1)
    f1_4 = f1_score(yy[m1], pm[m1].argmax(1), average="macro")
    cv["oof_f1_single_limb"] = float(f1_single); cv["oof_f1_4limb_mean"] = float(f1_4)
    print(f"OOF window F1 single-limb {f1_single:.4f} | 4-limb logmean {f1_4:.4f}", flush=True)
    np.save(os.path.join(out, "oof.npy"), oof); np.save(os.path.join(out, "test.npy"), test)
    json.dump(cv, open(os.path.join(out, "cv.json"), "w"), indent=1)
    pd.DataFrame({"id": tm.id, "target_feature": test.argmax(1)}).to_csv(os.path.join(out, "sub_raw_argmax.csv"), index=False)
    print("done", time.time() - t0)

if __name__ == "__main__":
    main()
