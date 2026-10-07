"""Shared helpers for the headroom/oracle study. OOF-only re-implementation of v4_local's single-fit decode
(--whiten 0.5 --whiten_mode subject --onehot, g1.5, tau 0.3, 8 members, 2 passes) + its boundary refiner (nested by
subject fold), with hooks to replace the links / the count targets / the probability masks by oracles."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
from v4_local import (H, macro_f1, N_CLS, TRAIN_SETS, FOLDS, profile_features, fit_counts, count_targets, finish_targets,
                      refiner_rows, refiner_flips, bag_P, load_fit, make_cfgs, whitened_subject, unit, REF_PARAMS, REF_ROUNDS, REF_THR)
from graph_lab import default_targets, calibrate_targets

HERE = os.path.dirname(os.path.abspath(__file__))
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
SUBS = r"E:\Claude code\wear\subs"
JOBS = 3
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def setup(path=K7):
    F = load_fit(path)
    y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    Bo = H.lsm(F["B2_OOF"].astype(np.float64)).astype(np.float32)
    lwo = H.lsm(F["oof_logp"].astype(np.float64)).astype(np.float32)
    Eo = unit(whitened_subject(F["oof_emb"], sbj, 0.5, 0.1))
    D = dict(F=F, y=y, sbj=sbj, fold=fold, Bo=Bo, lwo=lwo, Eo=Eo, rec=F["oof_rec"].astype(np.int64),
             start=F["oof_start"].astype(np.int64), ts=F["true_succ"].astype(np.int64),
             fold_of={int(s): int(f_) for s, f_ in zip(sbj, fold)}, cfg=make_cfgs(1.5, 0.3),
             Lo=[(F["oof_succ"][k], F["oof_score"][k]) for k in range(8)])
    D["order"] = time_order(D)
    return D


def time_order(D):
    """per recording, tile indices sorted by start (true time order)"""
    out = []
    for r in np.unique(D["rec"]):
        ii = np.flatnonzero(D["rec"] == r); out.append(ii[np.argsort(D["start"][ii], kind="stable")])
    return out


def dd_of(D, logp=None):
    return dict(logp=D["Bo"] if logp is None else logp, emb=D["Eo"], grp=D["sbj"], sbj=D["sbj"], sets=TRAIN_SETS)


def true_counts(D):
    y, sbj = D["y"], D["sbj"]
    return {int(s): np.bincount(y[sbj == s], minlength=N_CLS).astype(np.float64) for s in np.unique(sbj)}


def learned_counts(D, P):
    """nested (by subject fold) learned per-(subject, class) counts from profile features of P and the base blend"""
    y, sbj = D["y"], D["sbj"]
    X, key = profile_features([P, np.exp(D["Bo"].astype(np.float64))], sbj, TRAIN_SETS)
    X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1)
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    kf = np.array([D["fold_of"][int(s)] for s, _ in key]); cnt = np.zeros(len(true))
    for f_ in range(FOLDS):
        cnt[kf == f_] = fit_counts(X[kf != f_], true[kf != f_], X[kf == f_])
    return cnt, key, true


def targets_from(D, key, cnt):
    return count_targets(D["sbj"], TRAIN_SETS, key, cnt)


def decode(D, Lo, counts="learned", lam=1.0, logp=None, P0=None, verbose=True, mask=None):
    """2-pass decode. counts: 'learned' | 'true' | 'mix' (learned + lam*(true-learned)).
    mask: optional (n,19) 0/1 allowed-class mask applied to the graph output P before every count/Sinkhorn step.
    returns dict(P0, P, Q, fin, cnt_err)"""
    y, sbj, fold = D["y"], D["sbj"], D["fold"]
    dd = dd_of(D, logp); tg0 = default_targets(sbj, TRAIN_SETS)
    msk = (lambda P: P) if mask is None else (lambda P: (lambda R: R / R.sum(1, keepdims=True))(np.clip(P, 1e-12, None) * mask))
    if P0 is None:
        P0, _ = bag_P(dd, Lo, tg0, D["cfg"], JOBS)
    P = msk(P0); tg_true = true_counts(D); errs = []
    for k in range(2):
        cnt, key, true = learned_counts(D, P)
        if counts == "true":
            tg = tg_true; errs.append(0.0)
        else:
            if counts == "mix":
                cnt = cnt + lam * (true - cnt)
            tg = targets_from(D, key, cnt); errs.append(float(np.abs(cnt - true).mean()))
        if k == 0:
            P, _ = bag_P(dd, Lo, tg, D["cfg"], JOBS); P = msk(P)
    Q = finish_targets(P, sbj, tg); fin = Q.argmax(1)
    if verbose:
        log(f"  decode counts={counts} lam={lam}: count err {np.round(errs, 2)}; pre-refiner OOF {macro_f1(y, fin):.4f} per fold "
            + " ".join(f"{macro_f1(y[fold == f_], fin[fold == f_]):.4f}" for f_ in range(FOLDS)))
    return dict(P0=P0, P=P, Q=Q, fin=fin, tg=tg)


def refine_oof(D, fin, Lo, P, Q, Bo=None, lwo=None):
    """the kernel's boundary refiner, OOF only (LightGBM trained on the other 4 subject folds)"""
    import lightgbm as lgb
    from joblib import Parallel, delayed
    F = D["F"]; y, fold = D["y"], D["fold"]
    Bo = D["Bo"] if Bo is None else Bo; lwo = D["lwo"] if lwo is None else lwo
    REF = dict(REF_PARAMS, num_threads=2)
    ro = Parallel(n_jobs=JOBS)(delayed(refiner_rows)(fin, su, Bo, P, Q, lwo, F["sensor_oof"].astype(np.int64), *F["sc_o"]) for su, _ in Lo)
    X = np.concatenate([r[0] for r in ro]); tl = np.concatenate([r[1] for r in ro]); ot = np.concatenate([r[2] for r in ro])
    T = ((y[tl] == ot) & (y[tl] != fin[tl])).astype(int); Fr = fold[tl]; pr = np.zeros(len(T))
    for k in range(FOLDS):
        tr = Fr != k
        if (~tr).sum() == 0:
            continue
        pr[~tr] = lgb.train(REF, lgb.Dataset(X[tr], T[tr]), REF_ROUNDS).predict(X[~tr])
    ref, nf = refiner_flips(fin, tl, ot, pr, len(Lo))
    log(f"  refiner: {len(T)} rows, {nf} flipped, OOF {macro_f1(y, fin):.4f} -> {macro_f1(y, ref):.4f} per fold "
        + " ".join(f"{macro_f1(y[fold == f_], ref[fold == f_]):.4f}" for f_ in range(FOLDS)))
    return ref


def true_links(D, score=None):
    """8 copies of the true successor chain with a constant high score (the 90th percentile of the kernel's scores)"""
    ts = D["ts"]; sc0 = D["Lo"][0][1]
    s = np.percentile(sc0[D["Lo"][0][0] >= 0], 90) if score is None else score
    return [(ts.copy(), np.where(ts >= 0, s, -50.0).astype(np.float32)) for _ in range(8)]


def corrected_links(D, frac, seed=0):
    """every member: a random `frac` of its WRONG links (tile linked to a non-successor, or unlinked while a true
    successor exists) replaced by the true successor; the replaced link gets a score drawn from that member's scores of
    CORRECT links; any other tile that pointed at the same target is unlinked (keeps the matching 1:1)"""
    ts = D["ts"]; out = []; rng = np.random.default_rng(seed)
    for k, (su, sc) in enumerate(D["Lo"]):
        su, sc = su.copy(), sc.copy()
        ok = (su == ts) & (su >= 0); wrong = np.flatnonzero((ts >= 0) & (su != ts))
        pick = wrong[rng.random(len(wrong)) < frac]
        good_sc = sc[ok]
        # unlink tiles that currently point at a target that will be re-assigned
        tgt = np.zeros(len(su), bool); tgt[ts[pick]] = True
        clash = np.flatnonzero((su >= 0) & tgt[np.maximum(su, 0)]); su[clash] = -1; sc[clash] = -50.0
        su[pick] = ts[pick]; sc[pick] = rng.choice(good_sc, len(pick))
        out.append((su, sc.astype(np.float32)))
        if k == 0:
            m = su >= 0
            log(f"  corrected links frac {frac}: member0 linked {m.mean():.3f} exact {(su[m] == ts[m]).mean():.4f}")
    return out


def typed_links(D, kind):
    """every member: replace ONLY the wrong links of one kind by the true successor -- kind 'cross': links to a tile with
    another TRUE label (the harmful ones); 'same': wrong links that stay inside the same label. Clashing links unlinked."""
    ts, y = D["ts"], D["y"]; out = []
    for k, (su, sc) in enumerate(D["Lo"]):
        su, sc = su.copy(), sc.copy(); ok = (su == ts) & (su >= 0); good_sc = sc[ok]; rng = np.random.default_rng(k)
        wrong = (ts >= 0) & (su != ts) & (su >= 0)                   # unlinked tiles stay unlinked
        cross = wrong & (y[np.maximum(su, 0)] != y)
        pick = np.flatnonzero(cross if kind == "cross" else wrong & ~cross)
        tgt = np.zeros(len(su), bool); tgt[ts[pick]] = True
        clash = np.flatnonzero((su >= 0) & tgt[np.maximum(su, 0)]); clash = np.setdiff1d(clash, pick)
        su[clash] = -1; sc[clash] = -50.0; su[pick] = ts[pick]; sc[pick] = rng.choice(good_sc, len(pick))
        out.append((su, sc.astype(np.float32)))
        if k == 0:
            m = su >= 0
            log(f"  typed links {kind}: replaced {len(pick)}, unlinked {len(clash)}; member0 linked {m.mean():.3f} exact {(su[m] == ts[m]).mean():.4f} "
                f"cross-label {(y[su[m]] != y[m]).mean():.4f}")
    return out


def bouts(D):
    """true bouts: maximal runs of one label in true time order. returns bout id per tile, and boundary distance"""
    y = D["y"]; n = len(y); bid = np.full(n, -1); dist = np.full(n, 10 ** 6); b = 0
    for ii in D["order"]:
        lab = y[ii]; chg = np.r_[True, lab[1:] != lab[:-1]]; ids = np.cumsum(chg) - 1 + b; bid[ii] = ids; b = ids.max() + 1
        t = np.arange(len(ii)); cp = np.flatnonzero(chg[1:]) + 1          # first tile of a new bout
        if len(cp):
            pos = np.searchsorted(cp, t, side="right")
            d_left = np.where(pos > 0, t - cp[np.maximum(pos - 1, 0)] + 1, 10 ** 6)     # tiles since the last change (>=1)
            d_right = np.where(pos < len(cp), cp[np.minimum(pos, len(cp) - 1)] - t, 10 ** 6)  # tiles until the next change (>=1)
            dist[ii] = np.minimum(d_left, d_right)
    return bid, dist
