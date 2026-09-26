"""Combine the fold-0 expert predictions; report fold-0 single-limb macro-F1 for each expert, the log-blend grid and a
leave-one-subject-out LR stack; also the value of adding the deep blend to our base blend (bl_v3b_v1_f).
python blend_f0.py <vid_oof.npy> <imu_oof.npy> <probe_oof.npy>"""
import os, sys, itertools, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
PREP = r"E:\Claude code\wear\data\prep"; WORK = r"E:\Claude code\wear\work"; EXPD = r"E:\Claude code\wear\exp\deep"; NC = 19
m = pd.read_csv(os.path.join(PREP, "train_meta.csv")); N = len(m); y = m.y.to_numpy(); pur = m.pur.to_numpy(); sbj = m.sbj.to_numpy()
perm = np.random.RandomState(0).permutation(np.unique(sbj)); fo = {s: i % 5 for i, s in enumerate(perm)}; fold = np.array([fo[s] for s in sbj])
ref = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy"), mmap_mode="r"); refv = ~np.isnan(np.asarray(ref[:, :, 0])); rng5 = np.random.RandomState(5)
RL = np.array([rng5.choice(np.where(v)[0]) if v.any() else -1 for v in refv]); del ref
E = np.where((RL >= 0) & (pur >= 0.8) & (fold == 0))[0]; yE = y[E]; sE = sbj[E]; lE = RL[E]
VA = np.where(fold == 0)[0]
def pick(path):
    o = np.load(path).astype(np.float32)
    if o.shape[0] == len(VA):
        full = np.full((N, 4, NC), np.nan, np.float32); full[VA] = o; o = full
    P = o[E, lE]; return np.where(np.isnan(P[:, :1]), 1.0 / NC, P)
def lg(P): return np.log(np.clip(P, 1e-6, 1))
def f1(P): return f1_score(yE, P.argmax(1), average="macro")
names = ["vid", "imu", "probe"]; Ps = [pick(p) for p in sys.argv[1:4]]
for n, P in zip(names, Ps): print(f"{n:6s} fold0 single-limb F1 {f1(P):.4f}")
best = (-1, None); grid = np.round(np.arange(0, 1.01, 0.1), 2)
for w in itertools.product(grid, grid, grid):
    if abs(sum(w) - 1) > 1e-6: continue
    L = sum(wi * lg(P) for wi, P in zip(w, Ps)); f = f1(L)
    if f > best[0]: best = (f, w)
print("best log-blend", round(best[0], 4), "weights (vid,imu,probe)", best[1])
for w in [(0.5, 0.5, 0), (0.4, 0.4, 0.2), (0.45, 0.45, 0.1), (0.34, 0.33, 0.33)]:
    print("  log-blend", w, round(f1(sum(wi * lg(P) for wi, P in zip(w, Ps))), 4))
# LOSO LR stack on log-probs + limb one-hot
Xs = np.concatenate([lg(P) for P in Ps] + [np.eye(4)[lE]], 1); Q = np.zeros((len(E), NC))
for s in np.unique(sE):
    tr = sE != s; lr = LogisticRegression(C=0.1, max_iter=500).fit(Xs[tr], yE[tr])
    Q[~tr][:, :] = 0; Q[np.where(~tr)[0][:, None], lr.classes_[None, :]] = lr.predict_proba(Xs[~tr])
print("LOSO LR stack (C=0.1)", round(f1(Q), 4))
wb = best[1]; D = sum(wi * lg(P) for wi, P in zip(wb, Ps)); D = np.exp(D - D.max(1, keepdims=True)); D /= D.sum(1, keepdims=True)
print("per-class deep blend", np.round(f1_score(yE, D.argmax(1), average=None, labels=range(NC)), 2).tolist())
# value inside our blend
for bp in [os.path.join(r"E:\Claude code\wear\exp\base", "bl_v3b_v1_f", "oof.npy"), os.path.join(WORK, "lgbm_v1", "oof.npy"), os.path.join(r"E:\Claude code\wear\exp\base", "v3b", "oof.npy")]:
    try:
        B = pick(bp)
    except Exception as e:
        print(bp, e); continue
    print(os.path.basename(os.path.dirname(bp)), "alone", round(f1(B), 4), "| + deep log-blend w:",
          [(wd, round(f1((1 - wd) * lg(B) + wd * lg(D)), 4)) for wd in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)])
json.dump({"weights": list(map(float, wb)), "f1": best[0]}, open(os.path.join(EXPD, "blend_f0.json"), "w"))
