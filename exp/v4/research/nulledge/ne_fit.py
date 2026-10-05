"""Nested-by-fold null-edge specialist on top of the refined 2-fit fusion.
Candidates: tiles within 3 steps of a label change along any of the 16 matchings (optionally + chain ends / all).
Model: LightGBM on label-free features. Modes
  null    : target y == null (the hypothesis); act tile -> null if p > t1, null tile -> best non-null class of Q if p < t0
  aligned : target 'flip helps' (act tile: y != lab ; null tile: y == argmax non-null Q); flip if p > t1 / p > t0
Thresholds (t1, t0) are chosen per outer fold on inner-fold predictions of the 4 training folds only.
  python ne_fit.py --mode null --cand cand3 [--write]"""
import os, sys, argparse, time
import numpy as np
from ne_common import HERE, K7
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
from v4_local import macro_f1, N_CLS, FOLDS, write_sub
import lightgbm as lgb

T0 = time.time()
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=0)
ROUNDS = 400
T1_GRID = np.r_[np.arange(0.40, 0.96, 0.025), 2.0]       # 2.0 = never flip act -> null
T0_GRID_NULL = np.r_[-1.0, np.arange(0.05, 0.61, 0.025)]  # -1 = never flip null -> act (mode null: flip if p < t0)
T0_GRID_AL = np.r_[np.arange(0.40, 0.96, 0.025), 2.0]     # mode aligned: flip if p > t0


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def f1_fast(y, pred):
    return macro_f1(y, pred)


def apply(lab, qa, idx, p, t1, t0, mode):
    """labels after the specialist's flips on candidate tiles idx with probabilities p"""
    new = lab.copy(); cur = lab[idx]
    if mode == "null":
        a2n = (cur > 0) & (p > t1); n2a = (cur == 0) & (p < t0)
    else:
        a2n = (cur > 0) & (p > t1); n2a = (cur == 0) & (p > t0)
    new[idx[a2n]] = 0; new[idx[n2a]] = qa[idx[n2a]]
    return new


def choose(y, lab, qa, idx, p, mode, mask_eval):
    """grid search (t1, t0) maximising macro-F1 over the tiles of mask_eval (candidate idx all inside mask_eval)"""
    yy = y[mask_eval]; pos = np.full(len(y), -1); pos[np.flatnonzero(mask_eval)] = np.arange(mask_eval.sum())
    lab_e = lab[mask_eval]; qa_e = qa[mask_eval]; ie = pos[idx]
    t0g = T0_GRID_NULL if mode == "null" else T0_GRID_AL
    best = (f1_fast(yy, lab_e), 2.0, t0g[0] if mode == "null" else 2.0)
    # coordinate-wise is enough but grid is cheap
    for t1 in T1_GRID:
        for t0 in t0g:
            s = f1_fast(yy, apply(lab_e, qa_e, ie, p, t1, t0, mode))
            if s > best[0] + 1e-9:
                best = (s, t1, t0)
    return best


def target(y, lab, qa, idx, mode):
    if mode == "null":
        return (y[idx] == 0).astype(int)
    cur = lab[idx]
    return np.where(cur > 0, y[idx] != cur, y[idx] == qa[idx]).astype(int)


def train(X, t):
    return lgb.train(PARAMS, lgb.Dataset(X, t), ROUNDS)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="null"); ap.add_argument("--cand", default="cand3")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--drop", default="", help="comma list of feature-name prefixes to drop")
    a = ap.parse_args()
    z = np.load(os.path.join(HERE, "feats.npz")); b = np.load(os.path.join(HERE, "base_cache.npz"))
    st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
    y = st["oof_y"].astype(np.int64); fold = st["oof_fold"].astype(np.int64)
    lab, lab_t = b["ref_o"].astype(np.int64), b["ref_t"].astype(np.int64)
    qa, qat = z["qao"].astype(np.int64), z["qat"].astype(np.int64)
    names = [str(s) for s in z["names"]]; keep = np.ones(len(names), bool)
    for pfx in [s for s in a.drop.split(",") if s]:
        keep &= ~np.array([nm.startswith(pfx) for nm in names])
    Xo, Xt = z["Xo"][:, keep], z["Xt"][:, keep]
    if a.cand == "cand3":
        co, ct = z["cdo"] <= 4, z["cdt"] <= 4
    elif a.cand == "any":
        co = (z["cdo"] <= 4) | (z["efo"] <= 1) | (z["ebo"] <= 1); ct = (z["cdt"] <= 4) | (z["eft"] <= 1) | (z["ebt"] <= 1)
    else:
        co, ct = np.ones(len(y), bool), np.ones(len(lab_t), bool)
    idx_all = np.flatnonzero(co)
    log(f"mode {a.mode}, cand {a.cand}: {len(idx_all)} OOF candidates, {ct.sum()} test; {keep.sum()} features; "
        f"base refined F1 {macro_f1(y, lab):.4f}")
    new = lab.copy(); p_outer = np.zeros(len(idx_all)); chosen = []
    for k in range(FOLDS):
        tr_idx = idx_all[fold[idx_all] != k]; te_idx = idx_all[fold[idx_all] == k]
        p_in = np.zeros(len(tr_idx))
        for j in range(FOLDS):
            if j == k:
                continue
            a_ = fold[tr_idx] != j; b_ = ~a_
            m = train(Xo[tr_idx[a_]], target(y, lab, qa, tr_idx[a_], a.mode))
            p_in[b_] = m.predict(Xo[tr_idx[b_]])
        s, t1, t0 = choose(y, lab, qa, tr_idx, p_in, a.mode, fold != k)
        base_in = macro_f1(y[fold != k], lab[fold != k])
        m = train(Xo[tr_idx], target(y, lab, qa, tr_idx, a.mode))
        p = m.predict(Xo[te_idx]); p_outer[fold[idx_all] == k] = p
        new = apply(new, qa, te_idx, p, t1, t0, a.mode)
        chosen.append((t1, t0))
        mk = fold == k
        log(f"fold {k}: inner F1 {base_in:.4f} -> {s:.4f} at t1={t1:.3f} t0={t0:.3f}; outer {macro_f1(y[mk], lab[mk]):.4f} -> "
            f"{macro_f1(y[mk], new[mk]):.4f} ({(new[mk] != lab[mk]).sum()} flips, {((new[mk] != lab[mk]) & (new[mk] == y[mk])).sum()} right, "
            f"{((new[mk] != lab[mk]) & (lab[mk] == y[mk])).sum()} broke)")
    pf_b = [macro_f1(y[fold == k], lab[fold == k]) for k in range(FOLDS)]
    pf_n = [macro_f1(y[fold == k], new[fold == k]) for k in range(FOLDS)]
    imp = sum(n_ > b_ for n_, b_ in zip(pf_n, pf_b))
    tt = target(y, lab, qa, idx_all, a.mode)
    from sklearn.metrics import roc_auc_score
    log(f"RESULT mode {a.mode} cand {a.cand}: OOF {macro_f1(y, lab):.4f} -> {macro_f1(y, new):.4f}; per fold "
        + " ".join(f"{v:.4f}" for v in pf_n) + f"; folds improved {imp}/5; outer AUC {roc_auc_score(tt, p_outer):.4f}; flips {(new != lab).sum()}")
    tag = f"{a.mode}_{a.cand}" + (f"_drop{a.drop.replace(',', '+')}" if a.drop else "")
    np.savez(os.path.join(HERE, f"res_{tag}.npz"), new=new, p_outer=p_outer, idx=idx_all, pf_n=pf_n, pf_b=pf_b)
    if a.write:
        # test: model on every OOF candidate; thresholds from the 5-fold outer predictions over the full OOF
        s, t1, t0 = choose(y, lab, qa, idx_all, p_outer, a.mode, np.ones(len(y), bool))
        m = train(Xo[idx_all], tt); it = np.flatnonzero(ct)
        new_t = apply(lab_t, qat, it, m.predict(Xt[it]), t1, t0, a.mode)
        log(f"test: thresholds t1={t1:.3f} t0={t0:.3f} (full-OOF F1 at those {s:.4f}); {(new_t != lab_t).sum()} test flips "
            f"({((new_t == 0) & (lab_t > 0)).sum()} to null, {((new_t > 0) & (lab_t == 0)).sum()} to activity)")
        out = os.path.join(r"E:\Claude code\wear\subs", "sub_research_nulledge")
        write_sub(st["ids"], new_t, out + ".csv")
        np.save(out + "_labo.npy", new.astype(np.int8)); np.save(out + "_labt.npy", new_t.astype(np.int8))
        log(f"wrote {out}.csv")


if __name__ == "__main__":
    main()
