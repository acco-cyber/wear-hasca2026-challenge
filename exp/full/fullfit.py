"""Full-data refits (all 22 train subjects) for test prediction, several seeds averaged.
python fullfit.py v3b [--seeds 0,1,2] [--rounds 82]   -> exp/full/v3b_full/test.npy
python fullfit.py v1  [--seeds 0,1]   [--rounds 95]   -> exp/full/v1_full/test.npy"""
import os, sys, time, argparse
sys.path.insert(0, r"E:\Claude code\wear\exp\base"); sys.path.insert(0, r"E:\Claude code\wear\src")
from common import *
import lightgbm as lgb
OUT = r"E:\Claude code\wear\exp\full"

def fit_predict(X, y, w, Xte, params, rounds, seeds):
    P = 0
    for s in seeds:
        p = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s)
        m = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=rounds); P = P + m.predict(Xte) / len(seeds)
        print(f"  seed {s} done", flush=True)
    return P.astype(np.float32)

def undersample(y, rng, null_x=3.0):
    cnt = np.bincount(y, minlength=NC); keep_null = int(null_x * np.median(cnt[1:])); nidx = np.where(y == 0)[0]
    msk = np.ones(len(y), bool)
    if len(nidx) > keep_null: msk[rng.choice(nidx, len(nidx) - keep_null, replace=False)] = False
    return msk

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(",")]; t0 = time.time(); rng = np.random.RandomState(0)
    m = meta(); y = m.y.to_numpy(); pur = m.pur.to_numpy(); N = len(m)
    rng0 = np.random.RandomState(1); keep = np.zeros((N, 4), bool)
    for _ in range(2):
        pick = rng0.randint(0, 4, N); keep[np.arange(N), pick] = True
    if a.model == "v3b":
        from feats_v3 import train_blocks, test_blocks
        IM, VB = train_blocks(); IMt, VBt, tl = test_blocks()
        ok = ~np.isnan(IM[:, :, 0]) & (pur >= 0.8)[:, None]
        trs, trl = np.where(ok & keep); ytr = y[trs]; msk = undersample(ytr, rng); trs, trl, ytr = trs[msk], trl[msk], ytr[msk]
        X = np.concatenate([IM[trs, trl], np.eye(4, dtype=np.float32)[trl], VB[trs]], 1)
        Xte = np.concatenate([IMt, np.eye(4, dtype=np.float32)[tl], VBt], 1)
        params = dict(objective="multiclass", num_class=NC, learning_rate=0.15, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.2,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=63, num_threads=a.threads, verbose=-1)
        rounds = a.rounds or 82
    elif a.model == "v1":
        from train_lgbm import build_train, build_test
        X0, y0, sec, limb, sbj, pur0 = build_train(os.path.join(PREP, "train_rows_cache.npz")); Xte, tm = build_test()
        ok = ~np.isnan(X0[:, 0]) & (pur0 >= 0.8) & keep[sec, limb]
        X = X0[ok]; ytr = y0[ok]; msk = undersample(ytr, rng); X = X[msk]; ytr = ytr[msk]
        params = dict(objective="multiclass", num_class=NC, learning_rate=0.1, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.3,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=63, num_threads=a.threads, verbose=-1)
        rounds = a.rounds or 95
    cnt = np.bincount(ytr, minlength=NC).astype(np.float64); w = ((cnt.mean() / np.maximum(cnt, 1)) ** 0.5)[ytr]
    print(f"{a.model}: train rows {len(ytr)} feats {X.shape[1]} rounds {rounds} seeds {seeds} ({time.time()-t0:.0f}s)", flush=True)
    P = fit_predict(X, ytr, w, Xte, params, rounds, seeds)
    od = os.path.join(OUT, a.tag or f"{a.model}_full"); os.makedirs(od, exist_ok=True); np.save(os.path.join(od, "test.npy"), P)
    print("saved", od, f"argmax null {np.mean(P.argmax(1)==0):.3f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
