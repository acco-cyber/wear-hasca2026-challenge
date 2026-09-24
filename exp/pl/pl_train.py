"""Step 2: pseudo-label self-training of the v3b booster.
For fold k: train rows = v3b recipe (other folds, 2 limbs/sec, null 3x median, sqrt class weights)
          + pseudo rows of the fold-k sim sessions (their simulated limb, decoded label) at weight w_ps
          + pseudo rows of ALL test windows (e7 labels) at weight w_ps.
Fixed rounds = v3b best_iter[k] * round_mult (no early stopping: fold-k true labels are never used).
Writes exp/pl/<tag>/oof.npy (N,4,19), test.npy.
python pl_train.py <tag> --labels labels_v3bv1f.pkl --test_sub <csv> --w_ps 1.0 [--threads 6]"""
import os, sys, time, argparse, json, pickle
sys.path.insert(0, r"E:\Claude code\wear\exp\base")
from common import *
import lightgbm as lgb
from feats_v3 import train_blocks, test_blocks

PL = r"E:\Claude code\wear\exp\pl"
BEST_ITER = {0: 64, 1: 53, 2: 75, 3: 99, 4: 64}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--labels", default="labels_v3bv1f.pkl")
    ap.add_argument("--test_sub", default=r"E:\Claude code\wear\subs\sub_transductive_mrf4_knn_w6_d128.csv")
    ap.add_argument("--w_ps", type=float, default=1.0); ap.add_argument("--w_test", type=float, default=None)
    ap.add_argument("--round_mult", type=float, default=1.0); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); w_test = a.w_ps if a.w_test is None else a.w_test
    IM, VB = train_blocks(); IMt, VBt, tl = test_blocks()
    def rows(s, l): return np.concatenate([IM[s, l], np.eye(4, dtype=np.float32)[l], VB[s]], 1)
    Xte = np.concatenate([IMt, np.eye(4, dtype=np.float32)[tl], VBt], 1)
    m = meta(); y = m.y.to_numpy(); pur = m.pur.to_numpy(); N = len(m); fold = sec_fold()
    ok = ~np.isnan(IM[:, :, 0]) & (pur >= 0.8)[:, None]
    rng0 = np.random.RandomState(1); keep = np.zeros((N, 4), bool)
    for _ in range(2):
        pick = rng0.randint(0, 4, N); keep[np.arange(N), pick] = True
    L = pickle.load(open(os.path.join(PL, a.labels), "rb"))
    ps_sec, ps_limb, ps_lab = [], [], []
    for s, d in L.items():
        sec = d["a"] + np.arange(d["n"]); ps_sec.append(sec); ps_limb.append(d["limb"]); ps_lab.append(d["lab"])
    ps_sec = np.concatenate(ps_sec); ps_limb = np.concatenate(ps_limb); ps_lab = np.concatenate(ps_lab)
    valid_ps = ~np.isnan(IM[ps_sec, ps_limb, 0]); ps_sec, ps_limb, ps_lab = ps_sec[valid_ps], ps_limb[valid_ps], ps_lab[valid_ps]
    ps_fold = fold[ps_sec]
    te_lab = pd.read_csv(a.test_sub).sort_values("id").iloc[:, 1].to_numpy().astype(int)
    print(f"pseudo sim rows {len(ps_sec)}, test rows {len(te_lab)}, feats {Xte.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    params = dict(objective="multiclass", num_class=NC, metric="multi_logloss", learning_rate=0.15, num_leaves=31,
                  min_data_in_leaf=60, feature_fraction=0.2, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  max_bin=63, num_threads=a.threads, verbose=-1, seed=a.seed)
    rng = np.random.RandomState(0)
    oof = np.full((N, 4, NC), np.nan, np.float32); test = np.zeros((len(Xte), NC), np.float32)
    for k in range(5):
        trs, trl = np.where(ok & keep & (fold != k)[:, None]); ytr = y[trs]
        cnt = np.bincount(ytr, minlength=NC); keep_null = int(3.0 * np.median(cnt[1:])); nidx = np.where(ytr == 0)[0]
        if len(nidx) > keep_null:
            dr = rng.choice(nidx, len(nidx) - keep_null, replace=False); msk = np.ones(len(ytr), bool); msk[dr] = False
            trs, trl, ytr = trs[msk], trl[msk], ytr[msk]
        cnt2 = np.bincount(ytr, minlength=NC).astype(np.float64); cw = (cnt2.mean() / np.maximum(cnt2, 1)) ** 0.5
        pm = ps_fold == k
        Xtr = np.concatenate([rows(trs, trl), rows(ps_sec[pm], ps_limb[pm]), Xte], 0)
        ytr_all = np.concatenate([ytr, ps_lab[pm], te_lab])
        w = np.concatenate([cw[ytr], a.w_ps * cw[ps_lab[pm]], w_test * cw[te_lab]])
        nr = max(10, int(round(BEST_ITER[k] * a.round_mult)))
        mdl = lgb.train(params, lgb.Dataset(Xtr, ytr_all, weight=w), num_boost_round=nr)
        vas, val = np.where(ok & (fold == k)[:, None])
        oof[vas, val] = mdl.predict(rows(vas, val)); test += mdl.predict(Xte).astype(np.float32) / 5
        print(f"fold {k}: train {len(ytr)} + pseudo {pm.sum()} + test {len(te_lab)} rows, {nr} rounds ({time.time()-t0:.0f}s)", flush=True)
        del Xtr
    od = os.path.join(PL, a.tag); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), oof); np.save(os.path.join(od, "test.npy"), test)
    f = eval_single(oof, name=a.tag); json.dump(dict(args=vars(a), oof_f1_single=f), open(os.path.join(od, "res.json"), "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
