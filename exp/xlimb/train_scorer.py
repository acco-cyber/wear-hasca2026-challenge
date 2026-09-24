"""Fit cross-limb ridge models + augmented successor scorer on the scorer-fit sessions only.
python train_scorer.py <tag> [draws=3] [feature_set=all|old]"""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
from chain import FEAT_NAMES
import lightgbm as lgb

def draw_limbs(imu4, rng):
    n = len(imu4); limb = rng.randint(0, 4, n)
    bad = np.isnan(imu4[np.arange(n), limb]).any(axis=(1, 2))
    for i in np.where(bad)[0]:
        ok = [l for l in range(4) if not np.isnan(imu4[i, l]).any()]
        if ok: limb[i] = rng.choice(ok)
    return limb

if __name__ == "__main__":
    tag = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 3; fs = sys.argv[3] if len(sys.argv) > 3 else "all"
    t0 = time.time(); meta, imu, vid, sl = load_prep()
    xl = XLModels().fit(imu, sl, FIT); print("xl fit", f"{time.time()-t0:.0f}s", flush=True)
    rng = np.random.RandomState(123); Xs, Ys = [], []
    for s in FIT:
        a, b = sl[s]; n = b - a; V = np.asarray(vid[a:b], np.float32); W4 = np.asarray(imu[a:b], np.float32)
        for r in range(R):
            limb = draw_limbs(W4, rng); W = W4[np.arange(n), limb]
            cand, F = all_pair_features(V, W, limb, xl)
            lab = (cand == (np.arange(n)[:, None] + 1)).astype(np.int8); lab[-1] = 0
            F2 = F.reshape(-1, F.shape[-1]); L2 = lab.reshape(-1); ok = ~np.isnan(F2[:, 0])
            pos = np.where(ok & (L2 == 1))[0]; neg = np.where(ok & (L2 == 0))[0]
            keep = rng.choice(neg, min(len(neg), 30 * len(pos)), replace=False); sel = np.concatenate([pos, keep])
            Xs.append(F2[sel]); Ys.append(L2[sel])
            print(f"{s} draw {r}: pos {len(pos)}/{n-1} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(Ys); names = FEAT_NAMES + NEW_NAMES
    cols = list(range(len(FEAT_NAMES))) if fs == "old" else list(range(len(names)))
    if fs == "noraw": cols = [i for i, nm in enumerate(names) if not (nm.startswith("i_") or nm.startswith("j_"))]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=3)
    m.fit(X[:, cols], y, categorical_feature=[c for c in range(len(cols)) if names[cols[c]] in ("la", "lb")] if fs != "old" else "auto")
    imp = sorted(zip([names[c] for c in cols], m.booster_.feature_importance("gain")), key=lambda x: -x[1])
    print("importance(gain):", [(k, int(v)) for k, v in imp[:25]], flush=True)
    pickle.dump(dict(xl=xl, m=m, cols=cols, names=names), open(os.path.join(XD, f"scorer_{tag}.pkl"), "wb"))
    print("saved", f"{time.time()-t0:.0f}s")
