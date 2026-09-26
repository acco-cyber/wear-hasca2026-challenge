"""akhyar-style hierarchical LightGBM (family -> variant, + flat, 0.7/0.3, inverse-frequency weights, cap per
(subject,class), null x exp(0.75)) trained on the v3b FEATURES (mirror-canonical IMU + per-subject z-scored copy +
session-centred video). Full-data fit -> exp/full/akav3_full/test.npy.
python akastyle_v3.py [--rounds 400] [--lr 0.14] [--threads 4] [--seed 0] [--cap 600]"""
import os, sys, time, argparse
sys.path.insert(0, r"E:\Claude code\wear\exp\base")
from common import *
import lightgbm as lgb
from feats_v3 import train_blocks, test_blocks
FAMS = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12], [13, 14], [15], [16, 17], [18]]
FAM_OF = np.zeros(NC, int)
for f, cl in enumerate(FAMS):
    for c in cl: FAM_OF[c] = f

def invfreq(y):
    cnt = np.bincount(y); w = 1.0 / (cnt[y] + 1); return (w / w.mean()).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--lr", type=float, default=0.14)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--tag", default="akav3_full")
    a = ap.parse_args(); t0 = time.time(); rng = np.random.RandomState(a.seed)
    IM, VB = train_blocks(); IMt, VBt, tl = test_blocks(); m = meta(); y = m.y.to_numpy(); pur = m.pur.to_numpy(); sbj = m.sbj.to_numpy()
    sel = np.where(pur >= 0.999)[0]                                   # tiles fully inside a label segment
    keep = []
    for s in np.unique(sbj):
        for c in range(NC):
            idx = sel[(sbj[sel] == s) & (y[sel] == c)]
            if len(idx) > a.cap: idx = rng.choice(idx, a.cap, replace=False)
            keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    secs = np.repeat(keep, 4); limbs = np.tile(np.arange(4), len(keep))
    ok = ~np.isnan(IM[secs, limbs, 0]); secs, limbs = secs[ok], limbs[ok]
    X = np.concatenate([IM[secs, limbs], np.eye(4, dtype=np.float32)[limbs], VB[secs]], 1); Y = y[secs]
    Xte = np.concatenate([IMt, np.eye(4, dtype=np.float32)[tl], VBt], 1)
    print(f"rows {len(Y)} feats {X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    base = dict(learning_rate=a.lr, num_leaves=63, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8 * 0.5, lambda_l2=1.0,
                min_data_in_leaf=30, max_bin=63, num_threads=a.threads, verbose=-1, seed=a.seed)
    def fit(Xs, ys, k):
        p = dict(base, objective="multiclass", num_class=k)
        return lgb.train(p, lgb.Dataset(Xs, ys, weight=invfreq(ys)), num_boost_round=a.rounds)
    fam = FAM_OF[Y]; mf = fit(X, fam, len(FAMS)); Pf = mf.predict(Xte); print(f"  family model ({time.time()-t0:.0f}s)", flush=True)
    H = np.zeros((len(Xte), NC))
    for f, cl in enumerate(FAMS):
        if len(cl) == 1: H[:, cl[0]] = Pf[:, f]; continue
        msk = fam == f; yv = np.searchsorted(cl, Y[msk])
        mv = fit(X[msk], yv, len(cl)); Pv = mv.predict(Xte)
        H[:, cl] = Pf[:, [f]] * Pv; print(f"  variant fam{f} ({time.time()-t0:.0f}s)", flush=True)
    mflat = fit(X, Y, NC); Fl = mflat.predict(Xte); print(f"  flat ({time.time()-t0:.0f}s)", flush=True)
    P = 0.7 * H + 0.3 * Fl; P[:, 0] *= np.exp(0.75); P /= P.sum(1, keepdims=True)
    od = os.path.join(r"E:\Claude code\wear\exp\full", a.tag); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "test.npy"), P.astype(np.float32)); np.save(os.path.join(od, "test_hier.npy"), H.astype(np.float32)); np.save(os.path.join(od, "test_flat.npy"), Fl.astype(np.float32))
    print("saved", od, f"null {np.mean(P.argmax(1)==0):.3f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
