"""Train successor scorer (+ optional same-label pair model) on cached fit-session features; predict on the 18 sim sessions.
python train2.py <tag> [cols=all|old|noraw][+knn] [draws=0,1,2] [same=0|1] [seed=7]
-> models_<tag>.pkl, preds/<tag>_<session>.npz (lo, lq)"""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
from chain import FEAT_NAMES
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
FD = os.path.join(XD, "feats"); PD = os.path.join(XD, "preds"); os.makedirs(PD, exist_ok=True)
KNN_NAMES = ["kx_next_min", "kx_next_mean", "kx_same_min", "kx_same_mean"]
NAMES = FEAT_NAMES + NEW_NAMES + KNN_NAMES
NB = len(FEAT_NAMES) + len(NEW_NAMES)

def colset(fs):
    base, _, k = fs.partition("+")
    if base == "old": c = list(range(len(FEAT_NAMES)))
    elif base == "noraw": c = [i for i, nm in enumerate(NAMES[:NB]) if not (nm.startswith("i_") or nm.startswith("j_"))]
    else: c = list(range(NB))
    if k == "knn": c += list(range(NB, NB + len(KNN_NAMES)))
    return c

def load_F(s, r, knn):
    d = np.load(os.path.join(FD, f"{s}_{r}.npz")); F = d["F"]
    if knn: F = np.concatenate([F, np.load(os.path.join(FD, f"{s}_{r}_knn.npy"))], 2)
    else: F = np.concatenate([F, np.full(F.shape[:2] + (len(KNN_NAMES),), np.nan, np.float32)], 2)
    return d["cand"], F, d["y"]

def lgbm(n_est=600, leaves=63, mcs=100, seed=0):
    return lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, num_leaves=leaves, min_child_samples=mcs, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=3, random_state=seed)

def predict_lo(m, Z):
    sh = Z.shape[:-1]; Z2 = Z.reshape(-1, Z.shape[-1]); out = np.full(len(Z2), -50.0, np.float32); ok = ~np.isnan(Z2[:, 0])
    p = np.clip(m.predict_proba(Z2[ok])[:, 1], 1e-6, 1 - 1e-6); out[ok] = np.log(p / (1 - p)); return out.reshape(sh)

if __name__ == "__main__":
    tag = sys.argv[1]; fs = sys.argv[2] if len(sys.argv) > 2 else "all"
    draws = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
    do_same = int(sys.argv[4]) if len(sys.argv) > 4 else 0; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    knn = fs.endswith("+knn"); cols = colset(fs); rng = np.random.RandomState(seed); t0 = time.time()
    Xs, Ys, Xq, Yq = [], [], [], []
    for s in FIT:
        for r in draws:
            cand, F, y = load_F(s, r, knn); n = len(y)
            lab = (cand == (np.arange(n)[:, None] + 1)); lab[-1] = False
            v = cand >= 0; same = np.where(v, y[np.where(v, cand, 0)] == y[:, None], False)
            F2 = F.reshape(-1, F.shape[-1])[:, cols]; L2 = lab.reshape(-1); S2 = same.reshape(-1); ok = v.reshape(-1)
            pos = np.where(ok & L2)[0]; neg = np.where(ok & ~L2)[0]; keep = rng.choice(neg, min(len(neg), 30 * len(pos)), replace=False)
            sel = np.concatenate([pos, keep]); Xs.append(F2[sel]); Ys.append(L2[sel].astype(np.int8))
            if do_same:
                q = rng.choice(np.where(ok)[0], int(0.3 * ok.sum()), replace=False); Xq.append(F2[q]); Yq.append(S2[q].astype(np.int8))
    X, y = np.concatenate(Xs), np.concatenate(Ys)
    cat = [c for c in range(len(cols)) if NAMES[cols[c]] in ("la", "lb")] or "auto"
    print("train succ", X.shape, f"({time.time()-t0:.0f}s)", flush=True)
    ms = lgbm(seed=seed).fit(X, y, categorical_feature=cat); print("succ fit", f"({time.time()-t0:.0f}s)", flush=True)
    mq = lgbm(400, 63, 200, seed).fit(np.concatenate(Xq), np.concatenate(Yq), categorical_feature=cat) if do_same else None
    imp = sorted(zip([NAMES[c] for c in cols], ms.booster_.feature_importance("gain")), key=lambda x: -x[1])
    print("succ importance:", [(k, int(v)) for k, v in imp[:20]], flush=True)
    pickle.dump(dict(ms=ms, mq=mq, cols=cols, knn=knn), open(os.path.join(XD, f"models_{tag}.pkl"), "wb"))
    A = {}
    for w in SESS:
        for s in SESS[w]:
            cand, F, y = load_F(s, 0, knn); n = len(y); Z = F[..., cols]
            lo = predict_lo(ms, Z); lq = predict_lo(mq, Z) if mq is not None else np.zeros_like(lo); v = cand >= 0
            lab = (cand == (np.arange(n)[:, None] + 1)); A[s] = roc_auc_score(lab[v], lo[v])
            np.savez(os.path.join(PD, f"{tag}_{s}.npz"), lo=lo, lq=lq)
        print(w, "auc succ %.4f" % np.mean([A[s] for s in SESS[w]]), f"({time.time()-t0:.0f}s)", flush=True)
