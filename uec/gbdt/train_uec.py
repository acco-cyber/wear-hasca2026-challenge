"""UEC-dx2 Inertial-LightGBM member (approach_base/src/train_inertial_gbdt.py), manifest config:
  --model lightgbm --sensor-embedding-mode sensor --sensor-keys ra rl ll la --window-size 50 --stride 25
  --window-label-mode purity --min-label-purity 0.8 --normalization-mode none --smoothing-window 5
  --iterations 3000 --learning-rate 0.03 --depth 5 --num-leaves 31 --class-weight balanced
  --exclude-file-id-suffix-2  (+ repo defaults: early_stopping_rounds 100, feature_fraction 0.6, bagging 0.7/1,
  min_child_samples 50, min_split_gain 0.1, lambda_l1 0, lambda_l2 2, seed 42)

modes:
  manifest  5 contiguous subject folds over sorted subjects (run_subject_cv --num-folds 5), ES on the val fold
  std       our standard protocol folds (perm = RandomState(0).permutation(subjects), fold = i % 5), ES on val fold,
            OOF written as (69326, 4, 19) over data/prep/train_meta.csv (limbs left_arm, left_leg, right_arm, right_leg)
  full      all 22 sessions, fixed iterations (--iters), predicts test
usage: python train_uec.py manifest [fold ...] | std [fold ...] | full --iters N
"""
import os, sys, json, time, glob, argparse
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np

HERE = r"E:\Claude code\wear\uec\gbdt"
FEATS = os.path.join(HERE, "feats")
N_CLASSES = 19
THREADS = 4


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_all():
    files = sorted(p for p in glob.glob(os.path.join(FEATS, "sbj_*.npz")))
    Xs, M = [], {k: [] for k in ("start", "sensor", "y", "pur", "all4", "repo", "sbj", "session")}
    for p in files:
        d = np.load(p)
        sess = os.path.splitext(os.path.basename(p))[0]
        Xs.append(d["X"])
        for k in ("start", "sensor", "y", "pur", "all4", "repo", "sbj"):
            M[k].append(d[k])
        M["session"].append(np.full(len(d["y"]), sess))
    X = np.concatenate(Xs)
    del Xs
    M = {k: np.concatenate(v) for k, v in M.items()}
    M["y"] = M["y"].astype(np.int64)
    M["sbj"] = M["sbj"].astype(np.int64)
    return X, M


def load_test():
    d = np.load(os.path.join(FEATS, "test.npz"))
    assert (d["id"] == np.arange(len(d["id"]))).all()
    return d["X"]


def balanced_class_weights(y):
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    nz = counts > 0
    w = np.ones(N_CLASSES)
    w[nz] = len(y) / (float(nz.sum()) * counts[nz])
    return {int(i): float(w[i]) for i in np.where(nz)[0]}


def make_model(n_estimators, class_weight):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective="multiclass", num_class=N_CLASSES, n_estimators=n_estimators, learning_rate=0.03,
                          num_leaves=31, max_depth=5, random_state=42, class_weight=class_weight, n_jobs=THREADS,
                          verbosity=-1, device="cpu", feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
                          min_child_samples=50, min_split_gain=0.1, reg_alpha=0.0, reg_lambda=2.0)


def align(model, P):
    out = np.zeros((P.shape[0], N_CLASSES), np.float32)
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = P[:, j]
    return out


def fit_fold(X, M, tr, va, tag, outdir, Xtest, extra_pred=None):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, accuracy_score
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(os.path.join(outdir, "done.json")):
        log(tag, "already done")
        return json.load(open(os.path.join(outdir, "done.json")))
    ytr, yva = M["y"][tr], M["y"][va]
    cw = balanced_class_weights(ytr)
    log(tag, f"train={tr.sum()} val={va.sum()} val_sbj={sorted(set(M['sbj'][va].tolist()))}")
    model = make_model(3000, cw)
    t0 = time.time()
    model.fit(X[tr], ytr, eval_set=[(X[va], yva)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(50)])
    best = int(model.best_iteration_ or 3000)
    Pva = align(model, model.predict_proba(X[va]))
    f1 = float(f1_score(yva, Pva.argmax(1), average="macro", zero_division=0))
    acc = float(accuracy_score(yva, Pva.argmax(1)))
    log(tag, f"best_iter={best} val macro-F1={f1:.4f} acc={acc:.4f} fit {time.time() - t0:.0f}s")
    np.save(os.path.join(outdir, "val_prob.npy"), Pva)
    np.save(os.path.join(outdir, "val_idx.npy"), np.where(va)[0])
    np.save(os.path.join(outdir, "test_prob.npy"), align(model, model.predict_proba(Xtest)))
    if extra_pred is not None:
        np.save(os.path.join(outdir, "extra_prob.npy"), align(model, model.predict_proba(X[extra_pred])))
        np.save(os.path.join(outdir, "extra_idx.npy"), np.where(extra_pred)[0])
    model.booster_.save_model(os.path.join(outdir, "model.txt"))
    res = {"best_iter": best, "val_macro_f1": f1, "val_acc": acc, "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "val_subjects": sorted(set(M["sbj"][va].tolist())), "fit_seconds": time.time() - t0}
    json.dump(res, open(os.path.join(outdir, "done.json"), "w"), indent=1)
    return res


def contiguous_folds(subjects, k):
    base, rem = divmod(len(subjects), k)
    out, s = [], 0
    for i in range(k):
        e = s + base + (1 if i < rem else 0)
        out.append(subjects[s:e])
        s = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "std", "full"])
    ap.add_argument("folds", type=int, nargs="*")
    ap.add_argument("--iters", type=int, default=None)
    a = ap.parse_args()
    X, M = load_all()
    Xtest = load_test()
    log("loaded", X.shape, Xtest.shape)
    repo = M["repo"]
    subjects = sorted(set(M["sbj"][repo].tolist()))
    if a.mode == "manifest":
        folds = contiguous_folds(subjects, 5)
        for fi, vs in enumerate(folds):
            if a.folds and fi not in a.folds:
                continue
            va = repo & np.isin(M["sbj"], vs)
            tr = repo & ~np.isin(M["sbj"], vs)
            fit_fold(X, M, tr, va, f"manifest f{fi} {vs}", os.path.join(HERE, "manifest_cv", f"fold_{fi:02d}"), Xtest)
    elif a.mode == "std":
        import pandas as pd
        meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
        perm = np.random.RandomState(0).permutation(np.unique(meta.sbj))
        fold_of = {int(s): i % 5 for i, s in enumerate(perm)}
        for fi in range(5):
            if a.folds and fi not in a.folds:
                continue
            vs = [s for s, f in fold_of.items() if f == fi]
            va = repo & np.isin(M["sbj"], vs)
            tr = repo & ~np.isin(M["sbj"], vs)
            tiles = np.isin(M["sbj"], vs) & (M["start"] % 50 == 0)
            fit_fold(X, M, tr, va, f"std f{fi} {vs}", os.path.join(HERE, "std_cv", f"fold_{fi:02d}"), Xtest,
                     extra_pred=tiles)
    else:
        assert a.iters, "--iters required"
        outdir = os.path.join(HERE, "full")
        os.makedirs(outdir, exist_ok=True)
        tr = repo
        cw = balanced_class_weights(M["y"][tr])
        log(f"full fit rows={tr.sum()} iters={a.iters} subjects={subjects}")
        model = make_model(a.iters, cw)
        t0 = time.time()
        model.fit(X[tr], M["y"][tr])
        log(f"fit {time.time() - t0:.0f}s")
        P = align(model, model.predict_proba(Xtest))
        np.save(os.path.join(outdir, "test_prob.npy"), P)
        model.booster_.save_model(os.path.join(outdir, "model.txt"))
        json.dump({"iters": a.iters, "n_train": int(tr.sum()), "subjects": subjects, "fit_seconds": time.time() - t0},
                  open(os.path.join(outdir, "done.json"), "w"), indent=1)
        log("saved", P.shape)


if __name__ == "__main__":
    main()
