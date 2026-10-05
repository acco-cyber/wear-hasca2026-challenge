"""Null-edge specialist, v2: all nested fits in parallel (3 workers x 1 thread), predictions cached, then two ways of
using the nested null probability on top of the 2-fit fusion:
  flip : threshold flips on the refined labels (thresholds chosen on inner folds)          -- as in ne_fit.py
  sk   : the specialist's null odds pushed into the fused P before the count-constrained Sinkhorn (strength w chosen on
         inner folds, pre-refiner F1 of the training folds), then the boundary refiner re-run (nested by itself)
Feature sets: all | nosbj (drop subject-level context) | rank (+ within-subject percentile ranks of the evidence)
  python ne_fit2.py --feat nosbj --cand cand3 [--sk] [--write flip|sk]"""
import os, sys, argparse, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from joblib import Parallel, delayed
from ne_common import HERE, K7, load_all
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
import v4_combine as C
from v4_local import macro_f1, N_CLS, FOLDS, TRAIN_SETS, count_targets, finish_targets, write_sub, CFG, H
import ne_fit as NF

V.REF_PARAMS["num_threads"] = 2
T0 = time.time()
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1, seed=0)
ROUNDS = 250
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
SBJ_FEATS = ("sbj_", "lab_frac")
RANK_FEATS = ["P_lo", "Q_lo", "B_lo", "W_lo", "P7_lo", "P9_lo", "B7_lo", "B9_lo", "W7_lo", "W9_lo", "nbP_sum3", "ener", "vmot", "QB7_lo", "QB9_lo"]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def fit_pred(Xtr, ttr, Xte):
    import lightgbm as lgb
    return lgb.train(PARAMS, lgb.Dataset(Xtr, ttr), ROUNDS).predict(Xte)


def subj_rank(X, names, sb):
    out = []
    for nm in RANK_FEATS:
        v = X[:, names.index(nm)].astype(np.float64); r = np.zeros(len(v))
        for s in np.unique(sb):
            ii = np.flatnonzero(sb == s); o = np.argsort(np.argsort(v[ii], kind="stable"), kind="stable"); r[ii] = o / max(len(ii) - 1, 1)
        out.append(r)
    return np.stack(out, 1).astype(np.float32), [f"rk_{nm}" for nm in RANK_FEATS]


def lg(x):
    x = np.clip(x, 1e-4, 1 - 1e-4); return np.log(x) - np.log1p(-x)


def sk_labels(P, Q0, sbj, tg, idx, p, w, mask_sbj=None):
    """labels of the count-constrained finish after shifting the null odds of tiles idx towards the specialist's p"""
    P2 = P.copy(); T = CFG["sharpen_T"]
    P2[idx, 0] *= np.exp(T * w * (lg(p) - lg(np.clip(Q0[idx, 0], 1e-6, 1 - 1e-6))))
    P2 /= P2.sum(1, keepdims=True)
    if mask_sbj is None:
        return finish_targets(P2, sbj, tg).argmax(1)
    lab = np.full(len(P), -1); ii = np.flatnonzero(mask_sbj)
    lab[ii] = finish_targets(P2[ii], sbj[ii], tg).argmax(1)
    return lab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat", default="all"); ap.add_argument("--cand", default="cand3")
    ap.add_argument("--sk", action="store_true"); ap.add_argument("--write", default=""); ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    tag = f"{a.feat}_{a.cand}"
    z = np.load(os.path.join(HERE, "feats.npz")); b = np.load(os.path.join(HERE, "base_cache.npz"))
    st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
    y = st["oof_y"].astype(np.int64); fold = st["oof_fold"].astype(np.int64); sbj = st["oof_sbj"].astype(np.int64); tsbj = st["test_sbj"].astype(np.int64)
    lab, lab_t = b["ref_o"].astype(np.int64), b["ref_t"].astype(np.int64)
    qa, qat = z["qao"].astype(np.int64), z["qat"].astype(np.int64)
    names = [str(s) for s in z["names"]]; Xo, Xt = z["Xo"], z["Xt"]
    if a.feat in ("nosbj", "rank"):
        keep = ~np.array([nm.startswith(SBJ_FEATS) for nm in names]); Xo, Xt = Xo[:, keep], Xt[:, keep]
        names2 = [nm for nm, k_ in zip(names, keep) if k_]
        if a.feat == "rank":
            Ro, rn = subj_rank(z["Xo"], names, sbj); Rt, _ = subj_rank(z["Xt"], names, tsbj)
            Xo, Xt = np.concatenate([Xo, Ro], 1), np.concatenate([Xt, Rt], 1); names2 += rn
        names = names2
    if a.cand == "cand3":
        co, ct = z["cdo"] <= 4, z["cdt"] <= 4
    else:
        co, ct = np.ones(len(y), bool), np.ones(len(lab_t), bool)
    idx = np.flatnonzero(co); it = np.flatnonzero(ct)
    tt = (y[idx] == 0).astype(int); fi = fold[idx]
    pc = os.path.join(HERE, f"preds_{tag}.npz")
    if os.path.exists(pc):
        zz = np.load(pc); p_out, p_in, p_test = zz["p_out"], zz["p_in"], zz["p_test"]
        log(f"loaded cached predictions {pc}")
    else:
        jobs = []
        for k in range(FOLDS):
            jobs.append((k, -1, fi != k, fi == k))
            for j in range(FOLDS):
                if j != k:
                    jobs.append((k, j, (fi != k) & (fi != j), fi == j))
        log(f"feat {a.feat}, cand {a.cand}: {len(idx)} OOF candidates, {len(it)} test, {Xo.shape[1]} features; {len(jobs) + 1} fits")
        res = Parallel(n_jobs=a.jobs)(delayed(fit_pred)(Xo[idx[tr]], tt[tr], Xo[idx[te]]) for _, _, tr, te in jobs)
        p_test = fit_pred(Xo[idx], tt, Xt[it])
        p_out = np.zeros(len(idx)); p_in = np.zeros((FOLDS, len(idx)))      # p_in[k] = inner prediction with outer fold k held out
        for (k, j, tr, te), r in zip(jobs, res):
            if j < 0:
                p_out[te] = r
            else:
                p_in[k, te] = r
        np.savez(pc, p_out=p_out, p_in=p_in, p_test=p_test, idx=idx, it=it)
        log("fits done")
    from sklearn.metrics import roc_auc_score
    log(f"outer AUC {roc_auc_score(tt, p_out):.4f} (act tiles {roc_auc_score(tt[lab[idx] > 0], p_out[lab[idx] > 0]):.4f}, "
        f"null tiles {roc_auc_score(tt[lab[idx] == 0], p_out[lab[idx] == 0]):.4f})")
    pf_b = [macro_f1(y[fold == k], lab[fold == k]) for k in range(FOLDS)]
    # ---------------- flip mode
    new = lab.copy()
    for k in range(FOLDS):
        trm = fi != k
        s, t1, t0 = NF.choose(y, lab, qa, idx[trm], p_in[k, trm], "null", fold != k)
        new = NF.apply(new, qa, idx[~trm], p_out[~trm], t1, t0, "null")
        mk = fold == k
        log(f"flip fold {k}: inner {macro_f1(y[fold != k], lab[fold != k]):.4f}->{s:.4f} t1={t1:.3f} t0={t0:.3f}; outer {pf_b[k]:.4f} -> "
            f"{macro_f1(y[mk], new[mk]):.4f} ({(new[mk] != lab[mk]).sum()} flips)")
    pf = [macro_f1(y[fold == k], new[fold == k]) for k in range(FOLDS)]
    log(f"RESULT flip {tag}: OOF {macro_f1(y, lab):.4f} -> {macro_f1(y, new):.4f}; per fold " + " ".join(f"{v:.4f}" for v in pf)
        + f"; folds improved {sum(n_ > b_ for n_, b_ in zip(pf, pf_b))}/5")
    flip_o = new
    if a.write == "flip":
        s, t1, t0 = NF.choose(y, lab, qa, idx, p_out, "null", np.ones(len(y), bool))
        new_t = NF.apply(lab_t, qat, it, p_test, t1, t0, "null")
        log(f"test thresholds t1={t1:.3f} t0={t0:.3f}: {(new_t != lab_t).sum()} test flips")
        out = os.path.join(r"E:\Claude code\wear\subs", "sub_research_nulledge")
        write_sub(st["ids"], new_t, out + ".csv"); np.save(out + "_labo.npy", flip_o.astype(np.int8)); np.save(out + "_labt.npy", new_t.astype(np.int8))
        log(f"wrote {out}.csv")
    if not a.sk:
        return
    # ---------------- Sinkhorn mode
    fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    Po, Pt = b["Po"].astype(np.float64), b["Pt"].astype(np.float64)
    cnt, ctt, key, kt, true = C.counts_for([Po, np.exp(b["Bo"].astype(np.float64))], [Pt, np.exp(b["Bt"].astype(np.float64))], y, sbj, tsbj, fold_of, True)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ctt)
    Q0, Q0t = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt); fin = Q0.argmax(1); fin_t = Q0t.argmax(1)
    assert (fin == b["fin_o"]).all()
    fin2 = fin.copy(); ws = []
    for k in range(FOLDS):
        trm = fi != k; msk = fold != k; best = None
        for w in W_GRID:
            l2 = sk_labels(Po, Q0, sbj, tg, idx[trm], p_in[k, trm], w, msk)
            s = macro_f1(y[msk], l2[msk])
            if best is None or s > best[0] + 1e-9:
                best = (s, w)
        w = best[1]; ws.append(w); mk = fold == k
        l2 = sk_labels(Po, Q0, sbj, tg, idx[~trm], p_out[~trm], w, mk)
        fin2[mk] = l2[mk]
        log(f"sk fold {k}: inner pre-refiner {macro_f1(y[msk], fin[msk]):.4f}->{best[0]:.4f} at w={w}; outer {macro_f1(y[mk], fin[mk]):.4f} -> {macro_f1(y[mk], fin2[mk]):.4f}")
    log(f"sk pre-refiner: {macro_f1(y, fin):.4f} -> {macro_f1(y, fin2):.4f}; per fold "
        + " ".join(f"{macro_f1(y[fold == k], fin2[fold == k]):.4f}" for k in range(FOLDS)))
    # test: w on the full OOF from the outer predictions
    best = None
    for w in W_GRID:
        s = macro_f1(y, sk_labels(Po, Q0, sbj, tg, idx, p_out, w))
        if best is None or s > best[0] + 1e-9:
            best = (s, w)
    wt = best[1]; fin2_t = sk_labels(Pt, Q0t, tsbj, tgt, it, p_test, wt)
    log(f"sk test: w={wt} (full-OOF pre-refiner {best[0]:.4f}); {(fin2_t != fin_t).sum()} test tiles changed")
    d = load_all()
    Lo = [(d["Lo"][m], d["Lso"][m]) for m in range(len(d["Lo"]))]; Lt = [(d["Lt"][m], d["Lst"][m]) for m in range(len(d["Lt"]))]
    # the refiner's Q input: the re-calibrated soft labels are not kept per fold, use the baseline Q (labels changed only)
    sc_o = (d["oof_ener"], d["oof_post"], d["oof_vmot"], d["oof_vmean"].astype(np.float32))
    sc_t = (d["test_ener"], d["test_post"], d["test_vmot"], d["test_vmean"].astype(np.float32))
    ref_o, ref_t, _, _ = V.refine(fin2, fin2_t, Lo, Lt, b["Bo"].astype(np.float64), b["Bt"].astype(np.float64), Po, Pt, Q0, Q0t,
                                  b["lwo"].astype(np.float64), b["lwt"].astype(np.float64), d["sensor_oof"].astype(np.int64),
                                  d["sensor_test"].astype(np.int64), sc_o, sc_t, y, fold, 3, V.REF_THR)
    pf = [macro_f1(y[fold == k], ref_o[fold == k]) for k in range(FOLDS)]
    log(f"RESULT sk {tag}: refined OOF {macro_f1(y, lab):.4f} -> {macro_f1(y, ref_o):.4f}; per fold " + " ".join(f"{v:.4f}" for v in pf)
        + f"; folds improved {sum(n_ > b_ for n_, b_ in zip(pf, pf_b))}/5; ws {ws}")
    np.savez(os.path.join(HERE, f"sk_{tag}.npz"), fin2=fin2, fin2_t=fin2_t, ref_o=ref_o, ref_t=ref_t)
    if a.write == "sk":
        out = os.path.join(r"E:\Claude code\wear\subs", "sub_research_nulledge")
        write_sub(st["ids"], ref_t, out + ".csv"); np.save(out + "_labo.npy", ref_o.astype(np.int8)); np.save(out + "_labt.npy", ref_t.astype(np.int8))
        log(f"wrote {out}.csv; test agreement with baseline {np.mean(ref_t == lab_t):.4f}")


if __name__ == "__main__":
    main()
