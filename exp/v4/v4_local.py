"""Local (CPU) decode of the v4 "Learned Links + Counts" pipeline from the stage inputs kept by our Kaggle forks
(kaggle/v4_fork/patch_v4.py -> keep4/). One fit: reproduces the kernel's stage-B decode (kNN smoothing g1.5, bagged L3
links, learned count prior in two passes) and its boundary refiner. Several fits: a joint decode -- log-prob average of
the fits' stage-B blends, the union of every fit's bagged matchings, concatenated kNN embeddings, count profiles from
the joint and every single fit -- followed by the refiner along all matchings.
  python v4_local.py --fits DIR[,DIR...] --tag NAME [--whiten 0.5] [--g 1.5] [--no_refine] [--jobs 4] [--members 8]"""
import os, sys, re, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
W = r"E:\Claude code\wear"
sys.path.insert(0, os.path.join(W, "exp", "hyb"))
import hanbat_stack as H
from hanbat_stack import macro_f1, N_CLS, CFG, _norm_rows
from graph_lab import Runner2, calibrate_targets, default_targets
TRAIN_SETS = {0: 2, 14: 2}
FOLDS = 5
RK = [60, 70, 80, 90, 97, 105, 115, 125, 140]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ graph stage
def parse(name):
    m = re.fullmatch(r"g([\d.]+)/(abs|row)([\d.]+)/b(-?[\d.]+)/a([\d.]+)/r(\d)w([\d.]+)(?:/t([\d.]+))?", name)
    g, lm, beta, b, a, r, w, tau = m.groups()
    c = dict(emb_g=float(g), link_mode=lm, beta=float(beta), link_b=float(b), alpha=float(a), rounds=int(r), T=2.0, w=float(w))
    if tau:
        c["tau"] = float(tau)
    return c


def make_cfgs(g, tau):
    return [c.replace("g0.5/", f"g{g}/") + (f"/t{tau}" if tau else "") for c in H.TEST_CFGS]


def graph_P_one(dd, targets, cfg_names):
    Ps, runners = [], {}
    for c in sorted((parse(nm) for nm in cfg_names), key=lambda c: c["link_b"]):
        b = c.pop("link_b")
        if b not in runners:
            runners.clear(); runners[b] = Runner2(dd, targets, None, 0.0, b=b)
        Ps.append(runners[b](c).astype(np.float32))
    return _norm_rows(np.mean(Ps, 0))


def bag_P(base, links, targets, cfg_names, jobs):
    """one graph decode per matching -> geometric mean (and the per-matching decodes)"""
    Ps = Parallel(n_jobs=jobs)(delayed(graph_P_one)(dict(base, succ=su, score=sc), targets, cfg_names) for su, sc in links)
    G = np.exp(np.mean([np.log(np.clip(q, 1e-9, None)) for q in Ps], 0))
    return G / G.sum(1, keepdims=True), Ps


# ------------------------------------------------------------------ learned count prior
def profile_features(mats, sbj, sets):
    X, key = [], []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); ns = sets.get(int(s), 1)
        for c in range(1, N_CLS):
            f = []
            for M in mats:
                q = M[ii][:, c]
                sc = np.sort(np.log(q + 1e-6) - np.log(np.delete(M[ii], c, 1).max(1) + 1e-6))[::-1]
                f += [sc[min(int(r * ns), len(sc) - 1)] for r in RK]
                f += [q.sum() / ns, (M[ii].argmax(1) == c).sum() / ns, (sc > 0).sum() / ns, (sc > -1).sum() / ns, (sc > 1).sum() / ns]
            X.append(f + [len(ii) / ns]); key.append((int(s), c))
    return np.array(X, np.float64), np.array(key)


def ridge_cv(X, y, Xte, alphas=(1, 3, 10, 30, 100, 300, 1000)):
    """sklearn RidgeCV (efficient LOO, intercept) re-implemented in NumPy"""
    xm, ym = X.mean(0), y.mean(); Xc, yc = X - xm, y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); Uy = U.T @ yc; best = None
    for a in alphas:
        d = s ** 2 / (s ** 2 + a); pred = U @ (d * Uy); h = (U ** 2) @ d + 1.0 / len(y)
        loo = np.mean(((yc - pred) / (1 - h)) ** 2)
        if best is None or loo < best[0]:
            best = (loo, a)
    a = best[1]; coef = Vt.T @ ((s / (s ** 2 + a)) * Uy)
    return (Xte - xm) @ coef + ym


def fit_counts(Xtr, ttr, Xte):
    reg = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    return np.clip(ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd), 70, 135)


def count_targets(sbj, sets, key, cnt):
    out = {}
    for s in np.unique(sbj):
        n = int((sbj == s).sum()); per = cnt[key[:, 0] == s] * sets.get(int(s), 1)
        null = max(n - per.sum(), CFG["null_min"] * n); out[int(s)] = np.r_[null, per * (n - null) / per.sum()]
    return out


def finish_targets(P, sbj, targets):
    Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); return calibrate_targets(Q / Q.sum(1, keepdims=True), sbj, targets)


# ------------------------------------------------------------------ boundary refiner (kernel section 9, unchanged)
REF_H, REF_THR, REF_ROUNDS = 3, 0.5, 300
REF_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)


def refiner_rows(fin, succ, B, P, Q, lw, sens, ener, post, vmot, vmean):
    n = len(fin)
    prv = np.full(n, -1); prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
    lP, lQ = np.log(np.clip(P, 1e-9, None)), np.log(Q + 1e-9)

    def step(g, k):
        for _ in range(abs(k)):
            g = succ[g] if k > 0 else prv[g]
            if g < 0:
                return -1
        return g

    rows, tiles, others = [], [], []
    for i in np.flatnonzero((succ >= 0) & (fin != fin[np.maximum(succ, 0)])):
        j = succ[i]; A, Bc = fin[i], fin[j]
        for off in range(-REF_H + 1, REF_H + 1):
            g = step(i, off) if off <= 0 else step(j, off - 1)
            if g < 0:
                continue
            cur, oth = fin[g], (Bc if off <= 0 else A)
            if cur == oth:
                continue
            nxt, prv_ = succ[g], prv[g]
            f = [off, int(cur == 0), int(oth == 0), B[g, cur] - B[g, oth], lP[g, cur] - lP[g, oth], lw[g, cur] - lw[g, oth], lQ[g, cur] - lQ[g, oth],
                 B[g, 0], lP[g, 0], ener[g], vmot[g],
                 float(np.linalg.norm(vmean[g] - vmean[prv_])) if prv_ >= 0 else -1.0, float(np.linalg.norm(vmean[g] - vmean[nxt])) if nxt >= 0 else -1.0]
            for nb in (-2, -1, 1, 2):
                gj = step(g, nb)
                if gj >= 0:
                    f += [B[gj, cur] - B[gj, oth], lP[gj, cur] - lP[gj, oth], ener[gj] - ener[g], int(sens[gj] == sens[g]),
                          float(np.linalg.norm(post[gj] - post[g])) if sens[gj] == sens[g] else -1.0, int(fin[gj] == cur)]
                else:
                    f += [0, 0, 0, 0, -1.0, -1]
            rows.append(f); tiles.append(g); others.append(oth)
    return np.array(rows, np.float32), np.array(tiles, np.int64), np.array(others, np.int64)


def refiner_flips(fin, tiles, others, prob, K, thr=REF_THR):
    acc = {}
    for g, o, p_ in zip(tiles, others, prob):
        acc[(g, o)] = acc.get((g, o), 0.0) + p_ / K
    best = {}
    for (g, o), m in acc.items():
        if m > thr and m > best.get(g, (0, 0))[0]:
            best[g] = (m, o)
    new = fin.copy()
    for g, (m, o) in best.items():
        new[g] = o
    return new, len(best)


def refine(fin_o, fin_t, Lo, Lt, Bo, Bt, Po, Pt, Qo, Qt, lwo, lwt, so, st, sc_o, sc_t, y, fold, jobs, thr=REF_THR):
    import lightgbm as lgb
    ro = Parallel(n_jobs=jobs)(delayed(refiner_rows)(fin_o, su, Bo, Po, Qo, lwo, so, *sc_o) for su, _ in Lo)
    X = np.concatenate([r[0] for r in ro]); tl = np.concatenate([r[1] for r in ro]); ot = np.concatenate([r[2] for r in ro])
    T = ((y[tl] == ot) & (y[tl] != fin_o[tl])).astype(int); Fr = fold[tl]
    pr = np.zeros(len(T))
    for k in range(FOLDS):
        tr = Fr != k
        pr[~tr] = lgb.train(REF_PARAMS, lgb.Dataset(X[tr], T[tr]), REF_ROUNDS).predict(X[~tr])
    ref_o, nf = refiner_flips(fin_o, tl, ot, pr, len(Lo), thr)
    log(f"refiner: {len(T)} rows, {T.mean():.3f} should flip; {nf} OOF tiles flipped, F1 {macro_f1(y, fin_o):.4f} -> {macro_f1(y, ref_o):.4f} | per fold "
        + " ".join(f"{macro_f1(y[fold == f], ref_o[fold == f]):.4f}" for f in range(FOLDS)))
    mdl = lgb.train(REF_PARAMS, lgb.Dataset(X, T), REF_ROUNDS); del X
    rt = Parallel(n_jobs=jobs)(delayed(refiner_rows)(fin_t, su, Bt, Pt, Qt, lwt, st, *sc_t) for su, _ in Lt)
    Xt = np.concatenate([r[0] for r in rt]); tlt = np.concatenate([r[1] for r in rt]); ott = np.concatenate([r[2] for r in rt])
    ref_t, nft = refiner_flips(fin_t, tlt, ott, mdl.predict(Xt), len(Lt), thr)
    log(f"refiner: {nft} test tiles flipped")
    return ref_o, ref_t, (tl, ot, pr), (tlt, ott, mdl.predict(Xt))


# ------------------------------------------------------------------ fits
def load_fit(d):
    st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True); lk = np.load(os.path.join(d, "links.npz"))
    f = {k: st[k] for k in st.files}
    f.update(oof_succ=lk["oof_succ"].astype(np.int64), oof_score=lk["oof_score"].astype(np.float32),
             test_succ=lk["test_succ"].astype(np.int64), test_score=lk["test_score"].astype(np.float32),
             oof_emb=np.load(os.path.join(d, "oof_emb.npy")).astype(np.float32), test_emb=np.load(os.path.join(d, "test_emb.npy")).astype(np.float32))
    p = os.path.join(d, "tile_scalars.npz")
    if os.path.exists(p):
        z = np.load(p); f["sc_o"] = (z["oof_ener"], z["oof_post"], z["oof_vmot"], z["oof_vmean"].astype(np.float32))
        f["sc_t"] = (z["test_ener"], z["test_post"], z["test_vmot"], z["test_vmean"].astype(np.float32))
    f["name"] = os.path.basename(os.path.dirname(d.rstrip("\\/"))) if os.path.basename(d.rstrip("\\/")) == "keep4" else os.path.basename(d.rstrip("\\/"))
    return f


def whitened(Eo, so, Et, stt, power):
    def centre(E, sb):
        E = E.astype(np.float32).copy()
        for s in np.unique(sb):
            E[sb == s] -= E[sb == s].mean(0)
        return E
    Eo, Et = centre(Eo, so), centre(Et, stt)
    ev, U = np.linalg.eigh(np.cov(Eo[::3].T)); Wm = (U / np.maximum(ev, 1e-8) ** power).astype(np.float32)
    return Eo @ Wm, Et @ Wm


def whitened_subject(E, sb, power, shrink):
    """each subject's embedding centred and whitened with its OWN covariance (shrunk towards a scaled identity): the
    same label-free transform for training and test subjects, so a new camera does not meet training axes"""
    out = np.empty_like(E, dtype=np.float32)
    for s in np.unique(sb):
        ii = np.flatnonzero(sb == s); X = E[ii].astype(np.float64); X -= X.mean(0)
        C = np.cov(X.T); C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        ev, U = np.linalg.eigh(C); out[ii] = (X @ (U / np.maximum(ev, 1e-8) ** power)).astype(np.float32)
    return out


def unit(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)


def write_sub(ids, labels, path):
    """same row mapping as the kernel: sample_submission ids, left-merged with (test id, label)"""
    sample = pd.read_csv(os.path.join(W, "data", "sample_submission.csv")); id_col, target_col = sample.columns[:2]
    ids = np.asarray(ids)
    try:
        ids = ids.astype(sample[id_col].dtype)
    except (ValueError, TypeError):
        pass
    sub = sample[[id_col]].merge(pd.DataFrame({id_col: ids, target_col: np.asarray(labels).astype(int)}), on=id_col, how="left")
    assert len(sub) == len(sample) and sub[target_col].notna().all()
    sub[target_col] = sub[target_col].astype(int); sub.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--g", type=float, default=1.5); ap.add_argument("--whiten", type=float, default=0.0)
    ap.add_argument("--whiten_mode", default="train", choices=["train", "subject"]); ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=-1.0, help="kNN temperature (default 0.3 when whitened, else 0.1)")
    ap.add_argument("--members", type=int, default=8, help="matchings used per fit"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2); ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--blend_T", type=float, default=1.0, help="temperature of the averaged blend (joint decode)")
    ap.add_argument("--fit_prof", action="store_true", help="count profiles also from every single fit's own final P (needs --fitP)")
    ap.add_argument("--onehot", action="store_true", help="count regressor: add the exercise identity")
    ap.add_argument("--ref_thr", type=float, default=REF_THR)
    ap.add_argument("--links", default="", help="replacement links npz file(s) (oof_/test_ succ, score), comma-separated")
    ap.add_argument("--aux", default="", help="keep4 dir to borrow tile scalars + sensors from (fits without them)")
    a = ap.parse_args()
    fits = [load_fit(p) for p in a.fits.split(",")]
    if a.aux and "sc_o" not in fits[0]:
        X_ = load_fit(a.aux); assert (X_["oof_y"] == fits[0]["oof_y"]).all()
        fits[0].update(sc_o=X_["sc_o"], sc_t=X_["sc_t"], sensor_oof=X_["sensor_oof"], sensor_test=X_["sensor_test"])
    F = fits[0]; y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64)
    for f in fits[1:]:
        assert (f["oof_y"] == y).all() and (f["test_sbj"] == tsbj).all() and (f["oof_sbj"] == sbj).all()
    fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    log(f"{len(fits)} fits: " + ", ".join(f"{f['name']} (B2 tile F1 {macro_f1(y, f['B2_OOF'].argmax(1)):.4f}, kernel final {macro_f1(y, f['QB_OOF'].argmax(1)):.4f}"
                                          + (f", refined {macro_f1(y, f['ref_oof']):.4f})" if "ref_oof" in f else ")") for f in fits))
    # joint inputs
    Bo = H.lsm(np.mean([f["B2_OOF"].astype(np.float64) for f in fits], 0) / a.blend_T).astype(np.float32)
    Bt = H.lsm(np.mean([f["B2_TEST"].astype(np.float64) for f in fits], 0) / a.blend_T).astype(np.float32)
    lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0)).astype(np.float32)
    lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in fits], 0)).astype(np.float32)
    Eo, Et = [], []
    for f in fits:
        if a.whiten > 0 and a.whiten_mode == "subject":
            eo, et = whitened_subject(f["oof_emb"], sbj, a.whiten, a.shrink), whitened_subject(f["test_emb"], tsbj, a.whiten, a.shrink)
        else:
            eo, et = (whitened(f["oof_emb"], sbj, f["test_emb"], tsbj, a.whiten) if a.whiten > 0 else (f["oof_emb"], f["test_emb"]))
        Eo.append(unit(eo)); Et.append(unit(et))
    Eo, Et = np.concatenate(Eo, 1) / np.sqrt(len(fits)), np.concatenate(Et, 1) / np.sqrt(len(fits))
    Lo = [(f["oof_succ"][k], f["oof_score"][k]) for f in fits for k in range(min(a.members, len(f["oof_succ"])))]
    Lt = [(f["test_succ"][k], f["test_score"][k]) for f in fits for k in range(min(a.members, len(f["test_succ"])))]
    if a.links:                                          # replacement link sets (e.g. relink.py fused links)
        Lo, Lt = [], []
        for pth in a.links.split(","):
            z = np.load(pth)
            Lo += [(z["oof_succ"][k].astype(np.int64), z["oof_score"][k].astype(np.float32)) for k in range(len(z["oof_succ"]))]
            Lt += [(z["test_succ"][k].astype(np.int64), z["test_score"][k].astype(np.float32)) for k in range(len(z["test_succ"]))]
        ts = F.get("true_succ")
        if ts is not None:
            m = Lo[0][0] >= 0; log(f"replacement links: {len(Lo)} matchings, first exact successor {(Lo[0][0][m] == ts[m]).mean():.4f}")
    cfg_names = make_cfgs(a.g, a.tau if a.tau > 0 else (0.3 if a.whiten > 0 else None))
    log(f"joint blend tile F1 {macro_f1(y, Bo.argmax(1)):.4f}; {len(Lo)} matchings; configs {cfg_names[0]} ...")
    dd = dict(logp=Bo, emb=Eo, grp=sbj, sbj=sbj, sets=TRAIN_SETS); dt = dict(logp=Bt, emb=Et, grp=tsbj, sbj=tsbj, sets={})
    tg0, tgt0 = default_targets(sbj, TRAIN_SETS), default_targets(tsbj, {})
    Pt, _ = bag_P(dt, Lt, tgt0, cfg_names, a.jobs); log("test pass 0 done")
    P, _ = bag_P(dd, Lo, tg0, cfg_names, a.jobs)
    log(f"oof pass 0: bagged, fixed 97: F1 {macro_f1(y, finish_targets(P, sbj, tg0).argmax(1)):.4f}")
    Bp, Btp = np.exp(Bo.astype(np.float64)), np.exp(Bt.astype(np.float64))
    xo = [np.asarray(f["QB_OOF"], np.float64) for f in fits] if a.fit_prof else []
    xt = [np.asarray(f["QB_TEST"], np.float64) for f in fits] if a.fit_prof else []
    for k in range(a.passes):
        X, key = profile_features([P, Bp] + xo, sbj, TRAIN_SETS); Xt, kt = profile_features([Pt, Btp] + xt, tsbj, {})
        if a.onehot:
            X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1); Xt = np.concatenate([Xt, np.eye(N_CLS - 1)[kt[:, 1] - 1]], 1)
        true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
        kf = np.array([fold_of[int(s)] for s, _ in key]); cnt = np.zeros(len(true))
        for f_ in range(FOLDS):
            cnt[kf == f_] = fit_counts(X[kf != f_], true[kf != f_], X[kf == f_])
        ct = fit_counts(X, true, Xt)
        tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
        lab = finish_targets(P, sbj, tg).argmax(1)
        log(f"pass {k + 1}: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, lab):.4f} per fold "
            + " ".join(f"{macro_f1(y[fold == f_], lab[fold == f_]):.4f}" for f_ in range(FOLDS))
            + f"; test means {dict((int(s), round(float(ct[kt[:, 0] == s].mean()), 1)) for s in np.unique(tsbj))}")
        if k < a.passes - 1:
            P, _ = bag_P(dd, Lo, tg, cfg_names, a.jobs); Pt, _ = bag_P(dt, Lt, tgt, cfg_names, a.jobs)
    Qo, Qt = finish_targets(P, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    out = os.path.join(W, "subs", f"sub_v4l_{a.tag}")
    np.save(out + "_Qo.npy", Qo.astype(np.float32)); np.save(out + "_Qt.npy", Qt.astype(np.float32))
    np.save(out + "_Po.npy", P.astype(np.float32)); np.save(out + "_Pt.npy", Pt.astype(np.float32))
    ref_o, ref_t = fin_o, fin_t
    if not a.no_refine and "sc_o" in F:
        ref_o, ref_t, _, _ = refine(fin_o, fin_t, Lo, Lt, Bo, Bt, P, Pt, Qo, Qt, lwo, lwt, F["sensor_oof"].astype(np.int64),
                                    F["sensor_test"].astype(np.int64), F["sc_o"], F["sc_t"], y, fold, a.jobs, a.ref_thr)
    np.save(out + "_labo.npy", ref_o.astype(np.int8)); np.save(out + "_labt.npy", ref_t.astype(np.int8))
    write_sub(F["ids"], ref_t, out + ".csv")
    msg = "; ".join(f"agree {f['name']} {np.mean(ref_t == f['ref_test']):.4f}" for f in fits if "ref_test" in f)
    log(f"FINAL {a.tag}: OOF F1 {macro_f1(y, fin_o):.4f} -> refined {macro_f1(y, ref_o):.4f}; wrote {out}.csv; {msg}")


if __name__ == "__main__":
    main()
