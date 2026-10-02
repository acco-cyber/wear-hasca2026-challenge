"""Baseline (no self-training) on OOF: pseudo-labels = current recipe (prior .3, counts .3, gate .55:top2, L0 + our chain links)
on cv_base_oof_logp_b.  Saves pl_oof.npz and prints the baseline metrics per fold; also builds the feature caches."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

sm = load_meta(); y, fold, sbj = sm["y"], sm["fold"], sm["sbj"]
win, S3, T, LB = base_parts()
chk = tab_blend(win, S3, T); log("tab blend reconstruction max abs err", float(np.abs(chk - LB).max()))
log("S3 recovered F1", round(macro_f1(y, S3.argmax(1)), 4), "T F1", round(macro_f1(y, T.argmax(1)), 4), "tab F1", round(macro_f1(y, LB.argmax(1)), 4))
ours, has = ours_oof(len(y))
t0 = time.time(); dd = graph_dd_oof(LB, sm, extra=True)
lab, base, Q, P, g = recipe(dd, ours, has)
log(f"recipe (+xl) F1 {macro_f1(y, lab):.4f} per fold {[round(v, 4) for v in per_fold(y, lab, fold)]}  ({time.time() - t0:.0f}s) gated {g.mean():.4f}")
t0 = time.time(); dd0 = graph_dd_oof(LB, sm, extra=False)
lab0, _, _, _, _ = recipe(dd0, ours, has)
log(f"recipe (L0 only) F1 {macro_f1(y, lab0):.4f} per fold {[round(v, 4) for v in per_fold(y, lab0, fold)]}  ({time.time() - t0:.0f}s)")
t0 = time.time(); labp = plain_graph(LB, sm)
log(f"plain hanbat graph F1 {macro_f1(y, labp):.4f} per fold {[round(v, 4) for v in per_fold(y, labp, fold)]}  ({time.time() - t0:.0f}s)")
conf = Q.max(1)
np.savez(os.path.join(HERE, "pl_oof.npz"), lab=lab, base=base, conf=conf, gated=g, lab_l0=lab0, lab_plain=labp)
acc = lab == y
log(f"pseudo-label acc {acc.mean():.4f}; has rows {acc[has].mean():.4f}, no-has rows {acc[~has].mean():.4f}")
for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    m = conf >= thr; log(f"conf>={thr}: keep {m.mean():.3f} acc {acc[m].mean():.4f}")
for q in (0.3, 0.5, 0.7):
    m = np.zeros(len(y), bool)
    for s in np.unique(sbj):
        ii = sbj == s; m[ii] = conf[ii] >= np.quantile(conf[ii], q)
    log(f"per-subject top {1 - q:.0%}: acc {acc[m].mean():.4f}")
t0 = time.time(); X = features("oof"); log("oof features", X["S3"].shape, X["T"].shape, f"{time.time() - t0:.0f}s")
t0 = time.time(); X = features("test"); log("test features", X["S3"].shape, X["T"].shape, f"{time.time() - t0:.0f}s")
