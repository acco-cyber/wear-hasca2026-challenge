"""Baseline reproduction + oracle decomposition of the count prior on the K7+K9 fused P."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
from v4_local import macro_f1, N_CLS, TRAIN_SETS, FOLDS, profile_features, fit_counts, count_targets, finish_targets, CFG

Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.npz"))
y, sbj, fold, tsbj = Z["y"], Z["sbj"], Z["fold"], Z["tsbj"]
Po, Bpo = Z["Po"], Z["Bpo"]
fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
X, key = profile_features([Po, Bpo], sbj, TRAIN_SETS)
X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1)
true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
kf = np.array([fold_of[int(s)] for s, _ in key]); cnt = np.zeros(len(true))
for f_ in range(FOLDS):
    cnt[kf == f_] = fit_counts(X[kf != f_], true[kf != f_], X[kf == f_])
lab = finish_targets(Po, sbj, count_targets(sbj, TRAIN_SETS, key, cnt)).argmax(1)
pf = lambda lab: " ".join(f"{macro_f1(y[fold == f_], lab[fold == f_]):.4f}" for f_ in range(FOLDS))
print(f"baseline: count err {np.abs(cnt - true).mean():.2f}  F1 {macro_f1(y, lab):.4f} | {pf(lab)}")
ref = np.load(r"E:\Claude code\wear\subs\sub_v4c_b2_s7s9_Qo.npy").argmax(1)
print("agreement with sub_v4c_b2_s7s9 pre-refiner labels:", np.mean(ref == lab), "its F1", round(macro_f1(y, ref), 4))
print("true counts: mean %.1f sd %.1f min %.1f max %.1f; regular share %.3f" % (true.mean(), true.std(), true.min(), true.max(), ((true > 55) & (true < 150)).mean()))
print("irregular keys:", [(int(s), int(c), round(float(t), 1), round(float(p), 1)) for (s, c), t, p in zip(key, true, cnt) if not (55 < t < 150)])
# per-class mean / sd of true counts
for c in range(1, N_CLS):
    m = key[:, 1] == c
    print(f"  class {c:2d}: true mean {true[m].mean():6.1f} sd {true[m].std():5.1f} | pred err {np.abs(cnt[m] - true[m]).mean():5.2f}")
# subject level
for s in np.unique(sbj):
    m = key[:, 0] == s; ii = sbj == s; ns = TRAIN_SETS.get(int(s), 1)
    print(f"  sbj {s:2d} fold {fold_of[int(s)]}: n {ii.sum():5d} null share {np.mean(y[ii] == 0):.3f} (P {Po[ii, 0].mean():.3f}, argmax {np.mean(Po[ii].argmax(1) == 0):.3f}) "
          f"mean true {true[m].mean():6.1f} pred {cnt[m].mean():6.1f} err {np.abs(cnt[m] - true[m]).mean():5.2f} bias {np.mean(cnt[m] - true[m]):+6.2f}")
# oracles
def score(c_, null_override=None):
    tg = count_targets(sbj, TRAIN_SETS, key, c_)
    if null_override is not None:
        for s in tg:
            n = int((sbj == s).sum()); nl = null_override[s]; t = tg[s].copy(); t[1:] *= (n - nl) / t[1:].sum(); t[0] = nl; tg[s] = t
    l = finish_targets(Po, sbj, tg).argmax(1); return macro_f1(y, l), l
true_null = {int(s): float((y[sbj == s] == 0).sum()) for s in np.unique(sbj)}
for nm, c_, no in (("true counts", true, None), ("true counts + true null", true, true_null), ("pred counts + true null", cnt, true_null),
                   ("true shares, pred total", None, None)):
    if c_ is None:
        c_ = np.zeros_like(cnt)
        for s in np.unique(sbj):
            m = key[:, 0] == s; c_[m] = true[m] / true[m].sum() * cnt[m].sum()
    f, l = score(c_, no); print(f"oracle {nm}: F1 {f:.4f} | {pf(l)}")
# shrink baseline towards per-class training mean, all lambdas (diagnostic only, not nested)
for lam in (0.1, 0.2, 0.3, 0.5):
    cm = np.zeros_like(cnt)
    for f_ in range(FOLDS):
        tr = (kf != f_) & (true > 55) & (true < 150)
        for c in range(1, N_CLS):
            cm[(kf == f_) & (key[:, 1] == c)] = true[tr & (key[:, 1] == c)].mean()
    c2 = (1 - lam) * cnt + lam * cm; f, l = score(c2)
    print(f"diag shrink {lam}: err {np.abs(c2 - true).mean():.2f} F1 {f:.4f}")
