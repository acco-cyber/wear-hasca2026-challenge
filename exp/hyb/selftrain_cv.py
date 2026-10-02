"""Per-subject cross-fitted self-training (transductive subject adaptation).
For every subject: K-fold over its own windows; a LightGBM trained on the other folds' PSEUDO-labels (current pipeline
output) predicts the held-out fold -> subject-adapted log-probs A (no window sees its own pseudo-label).
  python selftrain_cv.py cv   <oof_P.npy> [--tag t] [--rounds 120] [--k 5]
       -> OOF F1 of A alone, and of finish(P * softmax(A)^w); saves work/hanbat/st_<tag>_oof_A.npy
  python selftrain_cv.py test <test_P.npy> <labels.csv> [--tag t]
       -> saves work/hanbat/st_<tag>_test_A.npy  (pseudo-labels from the CSV)"""
import os, sys, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, macro_f1, KEEP, HB, TRAIN_SETS, N_CLS, _norm_rows

PARAMS = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.1, num_leaves=15, min_data_in_leaf=10,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63, num_threads=6, verbose=-1)


def feats(split):
    f = np.load(os.path.join(KEEP, f"feat_{split}.npz"))
    emb = np.load(os.path.join(KEEP, f"{split}_emb.npy")).astype(np.float32)
    return f["imu"].astype(np.float32), f["vmot"].astype(np.float32), f["vpca"].astype(np.float32), emb


def subject_A(X, ypl, k, rounds, seed, w=None):
    n = len(ypl); A = np.zeros((n, N_CLS), np.float32); rng = np.random.default_rng(seed); fold = rng.integers(0, k, n)
    for j in range(k):
        tr, va = fold != j, fold == j
        ds = lgb.Dataset(X[tr], ypl[tr], weight=None if w is None else w[tr], params={"max_bin": 63})
        bst = lgb.train(dict(PARAMS, seed=seed + j), ds, rounds)
        A[va] = np.log(np.clip(bst.predict(X[va]), 1e-6, 1))
    return A


def build_X(imu, vmot, vpca, emb, ii, d_emb=32):
    E = emb[ii] / (np.linalg.norm(emb[ii], axis=1, keepdims=True) + 1e-6); E = E - E.mean(0)
    U, S, Vt = np.linalg.svd(E, full_matrices=False); Z = (U[:, :d_emb] * S[:d_emb]).astype(np.float32)
    return np.concatenate([imu[ii], vmot[ii], vpca[ii], Z], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("P"); ap.add_argument("labels", nargs="?")
    ap.add_argument("--tag", default="a"); ap.add_argument("--rounds", type=int, default=120); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--pl", default="", help="cv: pseudo-label .npy (their row order)")
    a = ap.parse_args(); t0 = time.time()
    P = np.load(a.P).astype(np.float64); P /= P.sum(1, keepdims=True)
    if a.mode == "cv":
        sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
        y, sbj, fold = sm["y"], sm["sbj"], sm["fold"]; imu, vmot, vpca, emb = feats("oof"); sets = TRAIN_SETS
        pl = np.load(a.pl).astype(np.int64) if a.pl else finish(P, dict(sbj=sbj, sets=sets))
    else:
        bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64); imu, vmot, vpca, emb = feats("test"); sets = {}
        pl = pd.read_csv(a.labels).sort_values("id").target_feature.to_numpy().astype(np.int64); y = None
    A = np.zeros((len(sbj), N_CLS), np.float32)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); X = build_X(imu, vmot, vpca, emb, ii)
        acc = 0
        for sd in range(a.seeds):
            acc = acc + subject_A(X, pl[ii], a.k, a.rounds, 100 * int(s) + sd)
        A[ii] = acc / a.seeds
        msg = f"sbj {s}: n={len(ii)} agree(A, pseudo) {np.mean(A[ii].argmax(1) == pl[ii]):.3f}"
        if y is not None:
            msg += f"  acc pseudo {np.mean(pl[ii] == y[ii]):.3f} -> A {np.mean(A[ii].argmax(1) == y[ii]):.3f}"
        print(msg, f"[{time.time() - t0:.0f}s]", flush=True)
    np.save(os.path.join(HB, f"st_{a.tag}_{'oof' if a.mode == 'cv' else 'test'}_A.npy"), A)
    if y is not None:
        print(f"baseline F1 {macro_f1(y, pl):.4f} | A alone F1 {macro_f1(y, A.argmax(1)):.4f}")
        SA = np.exp(A - A.max(1, keepdims=True)); SA /= SA.sum(1, keepdims=True)
        for w in (0.15, 0.25, 0.4, 0.6, 1.0):
            lab = finish(_norm_rows(P * SA ** w), dict(sbj=sbj, sets=sets))
            print(f"product w={w}: F1 {macro_f1(y, lab):.4f}  per fold " + " ".join(f"{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(5)))


if __name__ == "__main__":
    main()
