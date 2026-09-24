"""abhinavm2811 LightGBM recipe on cache/*.npz (see abh_prep.py).
python abh_train.py <jobs comma list: full | 0 | 1 | 2> [--nj 3] [--rounds 80] [--save_at 56,64,80]
  full : one model on ALL 277,304 (second,limb) rows, fixed rounds; saves test probs at each --save_at iteration
         -> abh_full/test_r{k}.npy ; abh_full/test.npy = r64 (primary).
  k    : notebook GroupKFold(3) fold k by sbj_id, 600 rounds max, early stopping 50 on the (weighted) val logloss,
         Dataset built on all rows then .subset() exactly like the notebook -> abh_cv3/fold{k}.npz
         (val rows, OOF probs (n,4,19) in OUR limb order, test probs, best_iter, fold F1).
Params = notebook: multiclass, lr .05, num_leaves 63, subsample .8 (inactive: bagging_freq 0), colsample_bytree .7,
reg_lambda 1, max_bin 63, force_col_wise; sample weights = sklearn compute_class_weight('balanced') on all rows."""
import os, sys, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("--nj", type=int, default=3)
ap.add_argument("--rounds", type=int, default=80); ap.add_argument("--save_at", default="56,64,80"); ap.add_argument("--primary", type=int, default=64)
A = ap.parse_args()
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = str(A.nj)
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
NC = 19
NB_ORDER = [2, 3, 1, 0]      # notebook location block order (right_arm, right_leg, left_leg, left_arm) as OUR limb index


def log(s, f=[None]):
    if f[0] is None: f[0] = open(os.path.join(HERE, "logs", f"train_{A.jobs.replace(',', '_')}.log"), "a")
    s = time.strftime("%H:%M:%S ") + s; print(s, flush=True); f[0].write(s + "\n"); f[0].flush()


def main():
    t0 = time.time()
    tr = np.load(os.path.join(CACHE, "train.npz")); te = np.load(os.path.join(CACHE, "test.npz"))
    Fi, V, y, sbj, nan = tr["Fi"], tr["V"], tr["y"], tr["sbj"], tr["nan"]
    N = len(y)
    # rows: 4 blocks (one per notebook location) of all seconds; video tiled; labels/groups tiled
    X = np.concatenate([np.concatenate([Fi[:, j], V], 1) for j in NB_ORDER], 0).astype(np.float32)
    del V
    Y = np.tile(y, 4); G = np.tile(sbj, 4)
    Xte = te["X"]
    cw = compute_class_weight("balanced", classes=np.arange(NC), y=Y)
    W = cw[Y].astype(np.float32)
    params = dict(objective="multiclass", num_class=NC, learning_rate=0.05, num_leaves=63, max_depth=-1, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=1.0, max_bin=63, force_col_wise=True, verbose=-1, num_threads=A.nj)
    log(f"[{A.jobs}] X {X.shape} Xte {Xte.shape} nj {A.nj} load {time.time()-t0:.0f}s")
    full = lgb.Dataset(X, label=Y, weight=W, params=params, free_raw_data=False); full.construct()
    log(f"dataset constructed {time.time()-t0:.0f}s")
    folds = list(GroupKFold(n_splits=3).split(X, Y, G))
    for job in A.jobs.split(","):
        t1 = time.time()
        if job == "full":
            od = os.path.join(HERE, "abh_full"); os.makedirs(od, exist_ok=True)
            it = [0]
            def cb(env):
                it[0] = env.iteration + 1
                if it[0] % 10 == 0: log(f"  full iter {it[0]} {time.time()-t1:.0f}s")
            bst = lgb.train(params, full, num_boost_round=A.rounds, callbacks=[cb])
            bst.save_model(os.path.join(od, "model.txt"))
            for k in [int(s) for s in A.save_at.split(",")]:
                p = bst.predict(Xte, num_iteration=k).astype(np.float32)
                np.save(os.path.join(od, f"test_r{k}.npy"), p)
                if k == A.primary: np.save(os.path.join(od, "test.npy"), p)
            log(f"[full] done {A.rounds} rounds {time.time()-t1:.0f}s")
        else:
            k = int(job); tri, vai = folds[k]
            vs = sorted(set(G[vai].tolist()))
            log(f"[fold {k}] val subjects {vs} train rows {len(tri)} val rows {len(vai)}")
            trs = full.subset(tri.astype(np.int32)); vas = full.subset(vai.astype(np.int32))
            def cb(env):
                if (env.iteration + 1) % 10 == 0: log(f"  fold {k} iter {env.iteration+1} {time.time()-t1:.0f}s")
            bst = lgb.train(params, trs, num_boost_round=600, valid_sets=[vas],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0), cb])
            pv = bst.predict(X[vai], num_iteration=bst.best_iteration)
            f1 = f1_score(Y[vai], pv.argmax(1), average="macro")
            pt = bst.predict(Xte, num_iteration=bst.best_iteration).astype(np.float32)
            # OOF in (second, OUR limb) layout
            rows = np.unique(vai % N)
            oof = np.full((N, 4, NC), np.nan, np.float32)
            blk = vai // N; sec = vai % N
            for b, j in enumerate(NB_ORDER):
                m = blk == b; oof[sec[m], j] = pv[m]
            od = os.path.join(HERE, "abh_cv3"); os.makedirs(od, exist_ok=True)
            np.savez(os.path.join(od, f"fold{k}.npz"), rows=rows, oof=oof[rows], test=pt, best_iter=bst.best_iteration, f1=f1,
                     val_idx=vai, pv=pv.astype(np.float32))
            log(f"[fold {k}] macro F1 {f1:.4f} best_iter {bst.best_iteration} {time.time()-t1:.0f}s")


if __name__ == "__main__":
    main()
