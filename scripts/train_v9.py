#!/usr/bin/env python3
"""WEAR v9 trainer v2 — fixed rounds + post-hoc iteration sweep.
Usage: python3 train_v9.py fold <k> [rounds] | pick <k> | pickall"""
import sys, os, time
import numpy as np

PROC = "/home/z/my-project/data/proc"
OUT = f"{PROC}/models"
os.makedirs(OUT, exist_ok=True)
NCLASS, NLIMB = 19, 4

def load(pseudo_w=0.0):
    F = np.load(f"{PROC}/train_limbfX.npy")
    if np.isnan(F).any():
        ok = ~np.isnan(F).any(1)
        med = np.nanmedian(F[ok], 0)
        idx = np.where(np.isnan(F))
        F[idx[0], idx[1]] = med[idx[1]]
    yw = np.load(f"{PROC}/train_y.npy").astype(np.int64)
    sbw = np.load(f"{PROC}/train_sbj.npy").astype(np.int64)
    knn = np.load(f"{PROC}/knn_tr.npy")
    Z = np.load(f"{PROC}/video_tr_pca.npy")
    n_win = len(yw)
    limb = np.tile(np.arange(NLIMB, dtype=np.int64), n_win)
    winof = np.repeat(np.arange(n_win, dtype=np.int64), NLIMB)
    V = np.concatenate([Z[winof], knn[winof]], 1).astype(np.float32)
    oh = np.eye(NLIMB, dtype=np.float32)[limb]
    X = np.concatenate([F, V, oh], 1).astype(np.float32)
    y = yw[winof]
    w = np.ones(len(y), np.float32)
    if pseudo_w > 0:
        Xte = np.load(f"{PROC}/test_X.npy").astype(np.float32)
        yte = np.load(f"{PROC}/pseudo_y.npy").astype(np.int64)
        X = np.concatenate([X, Xte], 0)
        y = np.concatenate([y, yte])
        w = np.concatenate([w, np.full(len(yte), pseudo_w, np.float32)])
        sb = np.concatenate([sbw[winof], np.full(len(yte), -1, np.int64)])
    else:
        sb = sbw[winof]
    return X, y, sb, w, n_win

def folds_by_subject(n_folds=4, seed=0):
    sbw = np.load(f"{PROC}/train_sbj.npy").astype(np.int64)
    sbjs = np.unique(sbw)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sbjs))
    assign = {sbjs[p]: i % n_folds for i, p in enumerate(perm)}
    return np.array([assign[s] for s in sbw])

def train_fold(k, rounds, pseudo_w=0.0):
    import lightgbm as lgb
    X, y, sb, w, n_win = load(pseudo_w)
    vfw = folds_by_subject()
    vf = np.repeat(vfw, NLIMB)
    winof = np.repeat(np.arange(n_win), NLIMB)
    is_pseudo = np.zeros(len(y), bool)
    if pseudo_w > 0:
        is_pseudo[len(y) - 12234:] = True
        vf = np.concatenate([vf, np.zeros(12234, np.int64)])
        winof = np.concatenate([winof, np.zeros(12234, np.int64)])
    tr = ((vf != k) & (winof % 2 == 0)) | is_pseudo
    va = (vf == k) & ~is_pseudo
    t0 = time.time()
    params = dict(objective="multiclass", num_class=NCLASS, metric="multi_logloss",
                  learning_rate=0.12, num_leaves=47, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1,
                  lambda_l2=2.0, max_bin=127, num_threads=2, verbose=-1, seed=k)
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr])
    model = lgb.train(params, dtr, num_boost_round=rounds,
                      callbacks=[lgb.log_evaluation(50)])
    model.save_model(f"{OUT}/s{pseudo_w}_m{k}.txt")
    print(f"fold {k}: trained {rounds} iters in {time.time()-t0:.0f}s (pw={pseudo_w})", flush=True)

def pick(k, tag=""):
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    X, y, sb, w, n_win = load()
    vfw = folds_by_subject()
    vf = np.repeat(vfw, NLIMB)
    winof = np.repeat(np.arange(n_win), NLIMB)
    va = vf == k
    model = lgb.Booster(model_file=f"{OUT}/{tag}m{k}.txt")
    n_rounds = model.num_trees() // NCLASS
    # sweep on a stratified subsample of va for speed
    rng = np.random.RandomState(0)
    va_idx = np.where(va)[0]
    sub = rng.choice(va_idx, min(30000, len(va_idx)), replace=False)
    ysub = y[sub]
    Xva_all = None
    best = (0, n_rounds)
    for r in range(25, n_rounds + 1, 25):
        p = model.predict(X[sub], num_iteration=r)
        f1 = f1_score(ysub, p.argmax(1), average="macro")
        if f1 > best[0]:
            best = (f1, r)
        print(f"  k={k} rounds={r} subF1={f1:.4f}", flush=True)
    f1, r = best
    print(f"fold {k}: best rounds={r} subF1={f1:.4f}", flush=True)
    p_va = model.predict(X[va], num_iteration=r)
    p_te = model.predict(np.load(f"{PROC}/test_X.npy"), num_iteration=r)
    full_f1 = f1_score(y[va], p_va.argmax(1), average="macro")
    print(f"fold {k}: FULL va macroF1={full_f1:.4f} @ rounds={r}", flush=True)
    np.save(f"{OUT}/{tag}p_va_{k}.npy", p_va.astype(np.float32))
    np.save(f"{OUT}/{tag}p_te_{k}.npy", p_te.astype(np.float32))
    np.save(f"{OUT}/{tag}va_idx_{k}.npy", np.where(va)[0])
    np.save(f"{OUT}/{tag}best_r_{k}.npy", np.array([r]))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fold":
        k = int(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        pw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        train_fold(k, rounds, pw)
    elif cmd == "pick":
        k = int(sys.argv[2])
        tag = sys.argv[3] if len(sys.argv) > 3 else ""
        pick(k, tag)
    elif cmd == "pickall":
        tag = sys.argv[2] if len(sys.argv) > 2 else ""
        for k in range(4):
            pick(k, tag)
