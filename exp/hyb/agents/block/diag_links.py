"""Link graph stats on OOF (L0 proxy) and test (L2): correctness, fragment lengths, how many bouts a fragment spans,
and block accuracy implied by the current P (argmax class block / mass ratio)."""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP, finish, TRAIN_SETS
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]
blk = np.zeros(19, int); blk[B1] = 1; blk[B2] = 2
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
y, rec, st, sbj = sm["y"], sm["rec"], sm["start"], sm["sbj"]
n = len(y)
l0 = np.load(os.path.join(KEEP, "links_L0.npz")); succ = l0["oof_succ"].astype(np.int64); score = l0["oof_score"]
# true successor
order = np.lexsort((st, rec)); true_succ = np.full(n, -1)
same = rec[order[1:]] == rec[order[:-1]]
true_succ[order[:-1][same]] = order[1:][same]
has = succ >= 0
print(f"OOF L0: links {has.mean():.3f} of rows; correct {np.mean(succ[has] == true_succ[has]):.4f}; "
      f"same-rec {np.mean(rec[succ[has]] == rec[has]):.4f}; |dt|<=5s {np.mean((rec[succ[has]] == rec[has]) & (np.abs(st[succ[has]] - st[has]) <= 250)):.4f}")
for q in (0.0, 0.25, 0.5, 0.75):
    thr = np.quantile(score[has], q); m = has & (score >= thr)
    print(f"  score>=q{q}: correct {np.mean(succ[m] == true_succ[m]):.4f}")


def fragments(succ):
    n = len(succ); prv = np.full(n, -1); prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
    lab = np.full(n, -1); fid = 0
    for s0 in np.flatnonzero(prv < 0):
        j = s0; cnt = 0
        while j >= 0 and lab[j] < 0 and cnt < n:
            lab[j] = fid; j = succ[j]; cnt += 1
        fid += 1
    rest = np.flatnonzero(lab < 0)   # cycles
    for j in rest:
        if lab[j] < 0:
            k = j
            while lab[k] < 0:
                lab[k] = fid; k = succ[k]
            fid += 1
    return lab


fr = fragments(succ); sz = np.bincount(fr)
print(f"fragments {len(sz)}, row-weighted median size {np.median(sz[fr]):.0f}, mean {sz[fr].mean():.0f}; size>=100 covers {np.mean(sz[fr] >= 100):.3f}")
# how many distinct activity classes/blocks per fragment
nb = []; mix = 0; tot = 0
for f in np.unique(fr):
    ii = np.flatnonzero(fr == f); a = y[ii][y[ii] > 0]
    if len(a) == 0:
        continue
    b = blk[a]; maj = max((b == 1).sum(), (b == 2).sum()); mix += len(a) - maj; tot += len(a)
    nb.append(len(np.unique(a)))
nb = np.array(nb)
print(f"fragments with activity: {len(nb)}; distinct classes per fragment: mean {nb.mean():.2f}, >=2: {np.mean(nb >= 2):.3f}; block-impurity {mix/tot:.4f}")
P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy").astype(np.float64); P /= P.sum(1, keepdims=True)
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
a = y > 0
print(f"block acc on activity windows: pred(finish) {np.mean(blk[pred[a]] == blk[y[a]]):.4f} (pred null {np.mean(pred[a]==0):.4f}); "
      f"mass-ratio {np.mean((P[a][:, B1].sum(1) > P[a][:, B2].sum(1)) == (blk[y[a]] == 1)):.4f}")
err = a & (pred != y)
print(f"activity errors {err.sum()}: cross-block {np.mean((pred[err] > 0) & (blk[pred[err]] != blk[y[err]])):.3f}, "
      f"to-null {np.mean(pred[err] == 0):.3f}, same-block {np.mean((pred[err] > 0) & (blk[pred[err]] == blk[y[err]])):.3f}")
nerr = (y == 0) & (pred != 0); print(f"null rows predicted as activity: {nerr.sum()}")
# per class cross-block error counts
cm = np.zeros((19, 19), int); np.add.at(cm, (y, pred), 1)
cross = [(i, j, cm[i, j]) for i in range(1, 19) for j in range(1, 19) if blk[i] != blk[j] and cm[i, j] >= 30]
print("largest cross-block confusions (true,pred,count):", sorted(cross, key=lambda x: -x[2])[:15])
same_ = [(i, j, cm[i, j]) for i in range(1, 19) for j in range(1, 19) if i != j and blk[i] == blk[j] and cm[i, j] >= 30]
print("largest same-block confusions:", sorted(same_, key=lambda x: -x[2])[:15])
# test links
l2 = np.load(os.path.join(KEEP, "links_L2_test.npz")); ts = l2["succ"].astype(np.int64)
frt = fragments(ts); szt = np.bincount(frt)
print(f"test L2: links {np.mean(ts >= 0):.3f}; fragments {len(szt)}, row-weighted median size {np.median(szt[frt]):.0f}")
