"""Transductive tabular expert: the Hanbat 'T' LightGBM (hand-crafted IMU + video features) retrained on labelled
train subjects PLUS the target subjects' pseudo-labelled windows, cross-fitted over the target windows.
  python adapt_tab.py cv   <pseudo_labels.npy> [--tag t] [--k 3] [--wt 2.0] [--rounds 113]
       per fold f: train = other folds (true y) + fold-f rows (pseudo, other cross-fit parts) -> predict the held-out part
       -> work/hanbat/ad_<tag>_oof.npy (log-probs); prints F1 alone
  python adapt_tab.py test <labels.csv> [--tag t] [--k 5]
       train = all OOF rows (true) + test rows (pseudo, cross-fit) -> work/hanbat/ad_<tag>_test.npy"""
import os, sys, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import macro_f1, KEEP, HB, N_CLS

PARAMS = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.08, num_leaves=31, min_data_in_leaf=80,
              feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63, num_threads=8, verbose=-1, seed=0)


def X_of(split):
    f = np.load(os.path.join(KEEP, f"feat_{split}.npz"))
    return np.concatenate([f["imu"], f["vmot"], f["vpca"]], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("labels"); ap.add_argument("--tag", default="a")
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--wt", type=float, default=2.0); ap.add_argument("--rounds", type=int, default=113)
    a = ap.parse_args(); t0 = time.time()
    Xo = X_of("oof"); sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    y, fold = sm["y"], sm["fold"]; rng = np.random.default_rng(0)
    if a.mode == "cv":
        pl = np.load(a.labels).astype(np.int64); out = np.zeros((len(y), N_CLS), np.float32)
        for f in range(5):
            te = np.flatnonzero(fold == f); tr = np.flatnonzero(fold != f); part = rng.integers(0, a.k, len(te))
            for j in range(a.k):
                a_tr = te[part != j]; a_va = te[part == j]
                X = np.concatenate([Xo[tr], Xo[a_tr]]); Y = np.concatenate([y[tr], pl[a_tr]])
                W = np.concatenate([np.ones(len(tr)), np.full(len(a_tr), a.wt)])
                bst = lgb.train(PARAMS, lgb.Dataset(X, Y, weight=W, params={"max_bin": 63}), a.rounds)
                out[a_va] = np.log(np.clip(bst.predict(Xo[a_va]), 1e-7, 1))
                print(f"fold {f} part {j}: adapted acc {np.mean(out[a_va].argmax(1) == y[a_va]):.4f} (pseudo acc {np.mean(pl[a_va] == y[a_va]):.4f}) [{time.time() - t0:.0f}s]", flush=True)
        np.save(os.path.join(HB, f"ad_{a.tag}_oof.npy"), out)
        print(f"adapted T alone: F1 {macro_f1(y, out.argmax(1)):.4f} (plain T OOF 0.6400; pseudo-labels F1 {macro_f1(y, pl):.4f})")
    else:
        Xt = X_of("test"); pl = pd.read_csv(a.labels).sort_values("id").target_feature.to_numpy().astype(np.int64)
        out = np.zeros((len(pl), N_CLS), np.float32); part = rng.integers(0, a.k, len(pl))
        for j in range(a.k):
            a_tr = np.flatnonzero(part != j); a_va = np.flatnonzero(part == j)
            X = np.concatenate([Xo, Xt[a_tr]]); Y = np.concatenate([y, pl[a_tr]]); W = np.concatenate([np.ones(len(y)), np.full(len(a_tr), a.wt)])
            bst = lgb.train(PARAMS, lgb.Dataset(X, Y, weight=W, params={"max_bin": 63}), a.rounds)
            out[a_va] = np.log(np.clip(bst.predict(Xt[a_va]), 1e-7, 1))
            print(f"part {j}: agreement with pseudo {np.mean(out[a_va].argmax(1) == pl[a_va]):.4f} [{time.time() - t0:.0f}s]", flush=True)
        np.save(os.path.join(HB, f"ad_{a.tag}_test.npy"), out); print("saved")


if __name__ == "__main__":
    main()
