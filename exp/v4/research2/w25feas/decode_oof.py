"""(c) OOF-only replica of v4_local.main (graph decode with bagged matchings, two passes of the nested learned count
prior with exercise identity, per-subject whitening 0.5, boundary refiner with nested folds) for a given OOF link set.
No test decode, no submission.  python decode_oof.py --fits K7[,K9] --links {own,true,<npz>} [--mixtrue 0.5] --tag T"""
import os, sys, time, argparse
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
from common import *
sys.path.insert(0, os.path.join(W, "exp", "v4"))
import v4_local as V
from v4_local import (bag_P, profile_features, fit_counts, count_targets, finish_targets, refiner_rows, refiner_flips, load_fit,
                      whitened_subject, unit, make_cfgs, TRAIN_SETS, FOLDS, REF_PARAMS, REF_ROUNDS, REF_THR)
from hanbat_stack import macro_f1, N_CLS
import hanbat_stack as H
from graph_lab import default_targets

ap = argparse.ArgumentParser(); ap.add_argument("--fits", default="K7"); ap.add_argument("--links", default="own"); ap.add_argument("--tag", required=True)
ap.add_argument("--mixtrue", type=float, default=0.0, help="replace this share of rows' links by the true successor (curve)")
ap.add_argument("--true_score", type=float, default=-1.0); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--members", type=int, default=8)
a = ap.parse_args(); T0 = time.time()
fits = [load_fit({"K7": K7, "K9": K9}[f]) for f in a.fits.split(",")]
F = fits[0]; y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64); ts = F["true_succ"].astype(np.int64)
fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
Bo = H.lsm(np.mean([f["B2_OOF"].astype(np.float64) for f in fits], 0)).astype(np.float32)
lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0)).astype(np.float32)
Eo = np.concatenate([unit(whitened_subject(f["oof_emb"], sbj, 0.5, 0.1)) for f in fits], 1) / np.sqrt(len(fits))
ref = np.load(os.path.join(K7, "links.npz"))["qn_ref"]
if a.links == "own":
    Lo = [(f["oof_succ"][k], f["oof_score"][k]) for f in fits for k in range(a.members)]
elif a.links == "true":
    sc = float(a.true_score if a.true_score > -1 else np.quantile(ref, 0.9))
    Lo = [(ts.copy(), np.where(ts >= 0, sc, -50.0).astype(np.float32)) for _ in range(a.members)]
else:
    z = np.load(a.links); Lo = [(z["oof_succ"][k].astype(np.int64), z["oof_score"][k].astype(np.float32)) for k in range(len(z["oof_succ"]))]
if a.mixtrue > 0:                                         # rows whose link is forced to the truth (score = row's own, else ref q0.9)
    rng = np.random.default_rng(7); m = (rng.random(len(ts)) < a.mixtrue) & (ts >= 0); Lo2 = []
    for su, sc in Lo:
        su = su.copy(); sc = sc.copy(); prev_owner = np.full(len(su), -1); prev_owner[su[su >= 0]] = np.flatnonzero(su >= 0)
        tgt = ts[m]; clash = prev_owner[tgt]; clash = clash[(clash >= 0) & ~m[np.maximum(clash, 0)]]
        su[clash] = -1; sc[clash] = -50.0                  # keep 1:1: drop other links pointing at a forced target
        su[m] = ts[m]; sc[m] = np.where(sc[m] > -49, sc[m], np.quantile(ref, 0.9)); Lo2.append((su, sc))
    Lo = Lo2
h = ts >= 0
print(f"[{time.time() - T0:.0f}s] {a.tag}: {len(Lo)} matchings, exact successor of the first {np.mean(Lo[0][0][h] == ts[h]):.4f}, "
      f"mean over matchings {np.mean([np.mean(su[h] == ts[h]) for su, _ in Lo]):.4f}", flush=True)
cfg = make_cfgs(1.5, 0.3)
dd = dict(logp=Bo, emb=Eo, grp=sbj, sbj=sbj, sets=TRAIN_SETS); tg0 = default_targets(sbj, TRAIN_SETS)
P, _ = bag_P(dd, Lo, tg0, cfg, a.jobs)
print(f"[{time.time() - T0:.0f}s] pass 0 (fixed targets) F1 {macro_f1(y, finish_targets(P, sbj, tg0).argmax(1)):.4f}", flush=True)
Bp = np.exp(Bo.astype(np.float64))
for k in range(2):
    X, key = profile_features([P, Bp], sbj, TRAIN_SETS); X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1)
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    kf = np.array([fold_of[int(s)] for s, _ in key]); cnt = np.zeros(len(true))
    for f_ in range(FOLDS):
        cnt[kf == f_] = fit_counts(X[kf != f_], true[kf != f_], X[kf == f_])
    tg = count_targets(sbj, TRAIN_SETS, key, cnt); lab = finish_targets(P, sbj, tg).argmax(1)
    print(f"[{time.time() - T0:.0f}s] pass {k + 1}: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, lab):.4f}", flush=True)
    if k == 0:
        P, _ = bag_P(dd, Lo, tg, cfg, a.jobs)
Qo = finish_targets(P, sbj, tg); fin = Qo.argmax(1)
# refiner (OOF part of v4_local.refine, nested by subject fold)
import lightgbm as lgb
so = F["sensor_oof"].astype(np.int64)
ro = Parallel(n_jobs=a.jobs)(delayed(refiner_rows)(fin, su, Bo, P, Qo, lwo, so, *F["sc_o"]) for su, _ in Lo)
Xr = np.concatenate([r[0] for r in ro]); tl = np.concatenate([r[1] for r in ro]); ot = np.concatenate([r[2] for r in ro])
T = ((y[tl] == ot) & (y[tl] != fin[tl])).astype(int); Fr = fold[tl]; pr = np.zeros(len(T))
prm = dict(REF_PARAMS, num_threads=3)
for k in range(FOLDS):
    tr = Fr != k
    pr[~tr] = lgb.train(prm, lgb.Dataset(Xr[tr], T[tr]), REF_ROUNDS).predict(Xr[~tr])
ref_o, nf = refiner_flips(fin, tl, ot, pr, len(Lo), REF_THR)
f1 = macro_f1(y, ref_o)
print(f"[{time.time() - T0:.0f}s] FINAL {a.tag}: OOF F1 {macro_f1(y, fin):.4f} -> refined {f1:.4f} | per fold "
      + " ".join(f"{macro_f1(y[fold == f_], ref_o[fold == f_]):.4f}" for f_ in range(FOLDS)), flush=True)
np.save(os.path.join(OUT, "cache", f"dec_{a.tag}_lab.npy"), ref_o.astype(np.int8))
