"""Two-stage count prior: stage 1 = a count regressor; its counts calibrate the fused P (Sinkhorn); stage 2 = a regressor on
profile features of that calibrated Q (+ stage-1 count, + stage-1 features). Strictly nested: for an evaluated fold k
every stage-1 count used as a stage-2 TRAINING feature comes from a model fitted without fold k and without its own fold.
python stack.py <FS1:model1[+...]> <model2> [stage2 feature mode: q|qx]"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import FOLDS, macro_f1, TRAIN_SETS, N_CLS, count_targets, finish_targets, profile_features
import make_counts as M


def stage1(D, mem, Xs, true, key, kf, pool, evals):
    """mean over members of: for each fold j in evals fit on pool minus j, predict j"""
    out = []
    for (fs, mdl), X in zip(mem, Xs):
        out.append(C.cv_predict(C.MODELS[mdl], X, true, key, kf, evals, pool))
    return np.mean(out, 0)


def qfeats(D, cnt, key, subjects):
    """profile features of the fused P calibrated with cnt, for the given subjects (rows in key order)"""
    sbj = D["sbj"]; m_t = np.isin(sbj, subjects); m_k = np.isin(key[:, 0], subjects)
    tg = count_targets(sbj[m_t], TRAIN_SETS, key[m_k], cnt[m_k]); Q = finish_targets(D["Po"][m_t], sbj[m_t], tg)
    Xq, kq = profile_features([Q], sbj[m_t], TRAIN_SETS)
    assert (kq == key[m_k]).all()
    return Xq


def stage2_X(Xq, cnt1, Xbase, mode):
    oh = Xbase[:, -(N_CLS - 1):]
    parts = [Xq, cnt1[:, None], oh] if mode == "q" else [Xq, cnt1[:, None], Xbase]
    return np.concatenate(parts, 1)


def outer_fold(D, mem, m2, mode, k):
    Xs = [M.built(D, fs)[0] for fs, _ in mem]; X0, key, Xt, kt, true, kf = M.built(D, mem[0][0])
    tr_f = [f for f in range(FOLDS) if f != k]
    c1 = np.full(len(true), np.nan)
    c1[np.isin(kf, tr_f)] = stage1(D, mem, Xs, true, key, kf, tr_f, tr_f)[np.isin(kf, tr_f)]    # inner (without k)
    c1[kf == k] = stage1(D, mem, Xs, true, key, kf, range(FOLDS), [k])[kf == k]                   # outer
    subs_tr = np.unique(key[np.isin(kf, tr_f), 0]); subs_te = np.unique(key[kf == k, 0])
    a_, b_ = qfeats(D, c1, key, subs_tr), qfeats(D, c1, key, subs_te)
    Xq = np.zeros((len(true), a_.shape[1])); Xq[np.isin(kf, tr_f)] = a_; Xq[kf == k] = b_
    X2 = stage2_X(Xq, c1, X0, mode); tr = np.isin(kf, tr_f); te = kf == k
    return k, c1, C.MODELS[m2](X2[tr], true[tr], key[tr], X2[te], key[te])


def run_cv(D, mem, m2, mode):
    X0, key, Xt, kt, true, kf = M.built(D, mem[0][0])
    res = Parallel(n_jobs=3)(delayed(outer_fold)(D, mem, m2, mode, k) for k in range(FOLDS))
    cnt2 = np.zeros(len(true)); cnt1 = np.zeros(len(true))
    for k, c1, p in res:
        cnt2[kf == k] = p; cnt1[kf == k] = c1[kf == k]
    return cnt1, cnt2


def run_test(D, mem, m2, mode):
    """stage 1 OOF over all 5 folds for the training rows, fit on all for test; stage 2 fit on all training rows"""
    Xs = [M.built(D, fs)[0] for fs, _ in mem]; X0, key, Xt, kt, true, kf = M.built(D, mem[0][0])
    Xts = [M.built(D, fs)[2] for fs, _ in mem]
    c1 = stage1(D, mem, Xs, true, key, kf, range(FOLDS), range(FOLDS))
    c1t = np.mean([C.MODELS[mdl](X, true, key, Xt_, kt) for (fs, mdl), X, Xt_ in zip(mem, Xs, Xts)], 0)
    Xq = qfeats(D, c1, key, np.unique(key[:, 0]))
    tg = count_targets(D["tsbj"], {}, kt, c1t); Qt = finish_targets(D["Pt"], D["tsbj"], tg)
    Xqt, kq = profile_features([Qt], D["tsbj"], {}); assert (kq == kt).all()
    X2, X2t = stage2_X(Xq, c1, X0, mode), stage2_X(Xqt, c1t, Xt, mode)
    return c1t, C.MODELS[m2](X2, true, key, X2t, kt)


if __name__ == "__main__":
    mem = [s.split(":") for s in sys.argv[1].split("+")]; m2 = sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "q"
    name = sys.argv[4] if len(sys.argv) > 4 else None
    D = C.load(); y, fold = D["y"], D["fold"]
    for fs, _ in mem:
        M.built(D, fs)
    t0 = time.time()
    X0, key, Xt, kt, true, kf = M.built(D, mem[0][0])
    cnt1, cnt2 = run_cv(D, mem, m2, mode)
    for nm, c in (("stage1", cnt1), ("stage2", cnt2), ("mean(1,2)", 0.5 * (cnt1 + cnt2))):
        f, lab = C.f1_from_counts(D, D["Po"], key, c)
        print(f"{sys.argv[1]} -> {m2} [{mode}] {nm:9s}: err {np.abs(c - true).mean():5.2f} cen {C.centred_err(c, true, key):5.2f} F1 {f:.4f} | "
              + " ".join(f"{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(FOLDS)), flush=True)
    if name:
        c1t, c2t = run_test(D, mem, m2, mode)
        np.savez(os.path.join(C.HERE, f"counts_{name}.npz"), cnt=cnt2, ct=c2t, key=key, kt=kt, true=true)
        np.savez(os.path.join(C.HERE, f"counts_{name}_s1.npz"), cnt=cnt1, ct=c1t, key=key, kt=kt, true=true)
        print(f"saved counts_{name}.npz; test mean stage1 {c1t.mean():.1f} stage2 {c2t.mean():.1f}")
    print(f"{time.time() - t0:.0f}s")
