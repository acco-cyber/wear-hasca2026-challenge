"""training-label artifacts: sbj_10's repeated block-B 3rd session (t>=2564, 1421 tiles) and sbj_2's 1-activity 3rd
session (t>=3445, 119 tiles) are performed exercises labelled null.  Does fitting the transferable components (count
regressor, boundary refiner) WITHOUT those tiles help?  K7 single fit, nested by subject fold as in the kernel.
Reported on all OOF tiles and on the clean tiles (artifacts excluded)."""
import os, sys
sys.path.insert(0, os.path.join(r"E:\Claude code\wear", "exp", "v4"))
from common import *
import v4_local as V
from v4_local import profile_features, fit_counts, count_targets, finish_targets, refiner_rows, refiner_flips, REF_PARAMS, REF_ROUNDS, REF_THR, TRAIN_SETS, N_CLS
from joblib import Parallel, delayed
import lightgbm as lgb

F = V.load_fit(K7)
y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64); tsbj = F["test_sbj"].astype(np.int64)
rec = F["oof_rec"]; t = F["oof_start"] // 50
art = ((rec == 3) & (t >= 2564)) | ((rec == 14) & (t >= 3445)); clean = ~art
fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
P, Pt = F["PB_OOF"].astype(np.float64), F["PB_TEST"].astype(np.float64)
Bp, Btp = np.exp(F["B2_OOF"].astype(np.float64)), np.exp(F["B2_TEST"].astype(np.float64))


def rep(tag, lab, labt=None):
    s = f"{tag}: OOF {f1(y, lab):.4f} | clean {f1(y[clean], lab[clean]):.4f} | per fold " + " ".join(f"{f1(y[fold == k], lab[fold == k]):.4f}" for k in range(5))
    if labt is not None:
        s += f" | test agree w/ kernel ref {np.mean(labt == F['ref_test']):.4f}"
    print(s, flush=True)


def counts(drop_art, onehot):
    """drop_art: the regressor's TRAINING rows use profile features computed without the artifact tiles; the evaluated
    (held-out) subject's features always come from all its tiles, exactly as a test subject's would"""
    keep = clean if drop_art else np.ones(len(y), bool)
    X, key = profile_features([P[keep], Bp[keep]], sbj[keep], TRAIN_SETS); Xt, kt = profile_features([Pt, Btp], tsbj, {})
    Xa, keya = profile_features([P, Bp], sbj, TRAIN_SETS); assert (keya == key).all()
    if onehot:
        oh_ = np.eye(N_CLS - 1)[key[:, 1] - 1]
        X = np.concatenate([X, oh_], 1); Xa = np.concatenate([Xa, oh_], 1); Xt = np.concatenate([Xt, np.eye(N_CLS - 1)[kt[:, 1] - 1]], 1)
    true = np.array([(y[keep & (sbj == s)] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    kf = np.array([fold_of[int(s)] for s, _ in key]); cnt = np.zeros(len(true))
    for f_ in range(5):
        cnt[kf == f_] = fit_counts(X[kf != f_], true[kf != f_], Xa[kf == f_])
    ct = fit_counts(X, true, Xt)
    return count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct), np.abs(cnt - true).mean()


rep("kernel QB argmax", F["QB_OOF"].argmax(1)); rep("kernel refined", F["ref_oof"], F["ref_test"])
res = {}
for onehot in (False, True):
    tg, tgt, err = counts(False, onehot); Q = finish_targets(P, sbj, tg)
    print(f"onehot={onehot}: count err {err:.2f}; agreement with kernel QB argmax {np.mean(Q.argmax(1) == F['QB_OOF'].argmax(1)):.4f}")
    rep(f"  local counts onehot={onehot}", Q.argmax(1))
oh = True
for drop in (False, True):
    tg, tgt, err = counts(drop, oh); Qo, Qt = finish_targets(P, sbj, tg), finish_targets(Pt, tsbj, tgt)
    res[drop] = (Qo, Qt); rep(f"counts drop_art={drop} (err {err:.2f}) Q argmax", Qo.argmax(1))

# refiner, with / without the artifact rows in training
Lo = [(F["oof_succ"][k], F["oof_score"][k]) for k in range(8)]; Lt = [(F["test_succ"][k], F["test_score"][k]) for k in range(8)]
so, st_ = F["sensor_oof"].astype(np.int64), F["sensor_test"].astype(np.int64)
Bo, Bt = F["B2_OOF"].astype(np.float32), F["B2_TEST"].astype(np.float32); lwo, lwt = F["oof_logp"], F["test_logp"]
for cdrop in (False, True):
    Qo, Qt = res[cdrop]; fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    ro = Parallel(n_jobs=3)(delayed(refiner_rows)(fin_o, su, Bo, P, Qo, lwo, so, *F["sc_o"]) for su, _ in Lo)
    X = np.concatenate([r[0] for r in ro]); tl = np.concatenate([r[1] for r in ro]); ot = np.concatenate([r[2] for r in ro])
    rt = Parallel(n_jobs=3)(delayed(refiner_rows)(fin_t, su, Bt, Pt, Qt, lwt, st_, *F["sc_t"]) for su, _ in Lt)
    Xt = np.concatenate([r[0] for r in rt]); tlt = np.concatenate([r[1] for r in rt]); ott = np.concatenate([r[2] for r in rt])
    T = ((y[tl] == ot) & (y[tl] != fin_o[tl])).astype(int); Fr = fold[tl]
    for seed in (0, 1, 2):
        for rdrop in (False, True):
            if cdrop and not rdrop and seed > 0:
                continue
            use = ~art[tl] if rdrop else np.ones(len(T), bool)
            prm = dict(REF_PARAMS, num_threads=2, seed=seed)
            pr = np.zeros(len(T))
            for k in range(5):
                tr = (Fr != k) & use
                pr[Fr == k] = lgb.train(prm, lgb.Dataset(X[tr], T[tr]), REF_ROUNDS).predict(X[Fr == k])
            ref_o, _ = refiner_flips(fin_o, tl, ot, pr, len(Lo))
            mdl = lgb.train(prm, lgb.Dataset(X[use], T[use]), REF_ROUNDS)
            ref_t, _ = refiner_flips(fin_t, tlt, ott, mdl.predict(Xt), len(Lt))
            rep(f"counts drop={cdrop} refiner drop={rdrop} seed {seed} ({use.sum()} of {len(T)} rows)", ref_o, ref_t)
            np.save(os.path.join(OUT, f"s09_K7_c{int(cdrop)}r{int(rdrop)}s{seed}_labo.npy"), ref_o.astype(np.int8))
            np.save(os.path.join(OUT, f"s09_K7_c{int(cdrop)}r{int(rdrop)}s{seed}_labt.npy"), ref_t.astype(np.int8))
