"""order ceilings: HMM/Viterbi along TRUE time order vs along link path covers (L0 OOF)"""
import numpy as np
from common import *
from chainlib import path_cover, viterbi, fb_smooth

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS); lq = np.log(np.clip(Q, 1e-9, 1))
bid = true_bouts(y, rec, st)
print("baseline", round(macro_f1(y, pred), 4))
true_paths = []
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r); true_paths.append(ii[np.argsort(st[ii])])


def run(paths, lam, mode="vit", src=lq):
    out = pred.copy()
    for p in paths:
        if len(p) < 2:
            continue
        L = src[p]
        out[p] = viterbi(L, lam) if mode == "vit" else fb_smooth(L, lam).argmax(1)
    return out


for lam in (2, 5, 10, 20, 40):
    o = run(true_paths, lam); print(f"TRUE order viterbi lam={lam}: F1 {macro_f1(y, o):.4f}", per_fold(y, o, fold))
for thr in (-9, 0.5, 1.5, 2.3, 3.0):
    paths = path_cover(succ, score, thr)
    L = np.array([len(p) for p in paths]); cov = L[L >= 2].sum() / n
    same_b = np.mean(np.concatenate([bid[p[1:]] == bid[p[:-1]] for p in paths if len(p) > 1]))
    plus1 = np.mean(np.concatenate([(st[p[1:]] - st[p[:-1]]) == 50 for p in paths if len(p) > 1]))
    print(f"\nthr {thr}: paths {len(paths)} covered {cov:.3f} len median(>=2) {np.median(L[L >= 2]):.0f} mean {L[L >= 2].mean():.1f} max {L.max()} | consecutive same-bout {same_b:.3f} exact+1 {plus1:.3f}")
    for lam in (2, 5, 10):
        o = run(paths, lam); f = macro_f1(y, o)
        o2 = run(paths, lam, "fb"); f2 = macro_f1(y, o2)
        print(f"   lam={lam}: viterbi F1 {f:.4f} {per_fold(y, o, fold)} | fb F1 {f2:.4f}")
