"""v3 booster: mirror-canonicalised IMU (+ per-session z-scored copy), session-centred + temporal video features.
Same 5-fold subject-disjoint split as work/lgbm_v1. Writes exp/base/<tag>/oof.npy (N,4,19), test.npy (12234,19), res.json.
python train_v3.py <tag> [--lr 0.1] [--limbs_per_sec 2] [--null_x 3] [--drop imuz,vidc,vidx] [--sim]"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
from feats_v3 import train_blocks, test_blocks

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag")
    ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--limbs_per_sec", type=int, default=2); ap.add_argument("--null_x", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=1500); ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--ff", type=float, default=0.25); ap.add_argument("--mdl", type=int, default=60)
    ap.add_argument("--drop", default=""); ap.add_argument("--sim", action="store_true"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fixed_rounds", type=int, default=0, help=">0: no early stopping, train this many rounds")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--task", default="flat", help="flat (19) | family (8) | binary (activity vs null)")
    a = ap.parse_args(); t0 = time.time()
    IM, VB = train_blocks(); IMt, VBt, tl = test_blocks()
    drop = set(a.drop.split(",")) if a.drop else set()
    im_cols = np.arange(220)
    if "imuz" in drop: im_cols = np.arange(110)
    vb_cols = np.arange(VB.shape[1])
    if "vidc" in drop: vb_cols = vb_cols[(vb_cols < 160) | (vb_cols >= 320)]
    if "vidx" in drop: vb_cols = vb_cols[vb_cols < 320 + 64 + 48]
    IM = IM[:, :, im_cols]; IMt = IMt[:, im_cols]; VB = VB[:, vb_cols]; VBt = VBt[:, vb_cols]
    def rows(s, l): return np.concatenate([IM[s, l], np.eye(4, dtype=np.float32)[l], VB[s]], 1)
    Xte = np.concatenate([IMt, np.eye(4, dtype=np.float32)[tl], VBt], 1)
    m = meta(); y = m.y.to_numpy(); pur = m.pur.to_numpy(); N = len(m); fold = sec_fold()
    y19 = y.copy()
    if a.task == "family": y = FAMILY[y]
    if a.task == "binary": y = (y > 0).astype(np.int64)
    K = {"flat": NC, "family": 8, "binary": 2}[a.task]
    ok = ~np.isnan(IM[:, :, 0]) & (pur >= 0.8)[:, None]                     # (N,4)
    # training thinning: limbs_per_sec random limbs per second (same RNG recipe as train_lgbm.py)
    rng0 = np.random.RandomState(1 + 17 * a.seed); keep = np.zeros((N, 4), bool)
    for _ in range(a.limbs_per_sec):
        pick = rng0.randint(0, 4, N); keep[np.arange(N), pick] = True
    if a.limbs_per_sec >= 4: keep[:] = True
    print("feats", Xte.shape[1], "usable", ok.sum(), "train-thinned", (ok & keep).sum(), f"{time.time()-t0:.0f}s", flush=True)
    params = dict(objective="multiclass", num_class=K, metric="multi_logloss", learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=a.mdl, feature_fraction=a.ff, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  max_bin=63, num_threads=a.threads, verbose=-1, seed=a.seed)
    if K == 2: params.update(objective="binary", metric="binary_logloss"); params.pop("num_class")
    def predict(mdl, X, bi):
        p = mdl.predict(X, num_iteration=bi)
        return np.stack([1 - p, p], 1) if K == 2 else p
    rng = np.random.RandomState(a.seed)
    oof = np.full((N, 4, K), np.nan, np.float32); test = np.zeros((len(Xte), K), np.float32); cv = {}
    for k in range(5):
        trs, trl = np.where(ok & keep & (fold != k)[:, None])
        vas, val = np.where(ok & (fold == k)[:, None])
        ytr = y[trs]
        cnt = np.bincount(ytr, minlength=K); keep_null = int(a.null_x * np.median(cnt[1:]))
        nidx = np.where(ytr == 0)[0]
        if len(nidx) > keep_null:
            drop_i = rng.choice(nidx, len(nidx) - keep_null, replace=False); msk = np.ones(len(ytr), bool); msk[drop_i] = False
            trs, trl, ytr = trs[msk], trl[msk], ytr[msk]
        cnt2 = np.bincount(ytr, minlength=K).astype(np.float64); w = ((cnt2.mean() / np.maximum(cnt2, 1)) ** 0.5)[ytr]
        Xtr = rows(trs, trl); Xva = rows(vas, val)
        dtr = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=True)
        def cb(env):
            if env.iteration % 25 == 0:
                r = env.evaluation_result_list; print(f"    it {env.iteration} {r[0][2]:.4f} ({time.time()-t0:.0f}s)" if r else f"    it {env.iteration} ({time.time()-t0:.0f}s)", flush=True)
        if a.fixed_rounds:
            mdl = lgb.train(params, dtr, num_boost_round=a.fixed_rounds, callbacks=[cb]); bi = a.fixed_rounds
        else:
            dva = lgb.Dataset(Xva, y[vas], reference=dtr)
            mdl = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(a.patience, verbose=False), cb])
            bi = mdl.best_iteration
        p = predict(mdl, Xva, bi); oof[vas, val] = p
        test += predict(mdl, Xte, bi).astype(np.float32) / 5
        from sklearn.metrics import f1_score
        f1r = f1_score(y[vas], p.argmax(1), average="macro")
        cv[k] = dict(best_iter=int(bi), f1_rows=float(f1r), n_tr=int(len(ytr)))
        print(f"fold {k}: iter {bi} row-F1 {f1r:.4f} n_tr {len(ytr)} ({time.time()-t0:.0f}s)", flush=True)
        del Xtr, Xva, dtr
    od = os.path.join(EXP, a.tag); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), oof); np.save(os.path.join(od, "test.npy"), test)
    cv["args"] = vars(a)
    if K == NC: cv["oof_f1_single"] = eval_single(oof, name=a.tag)
    if a.sim and K == NC:
        s, _ = run_sim(os.path.join(od, "oof.npy"), a.tag); cv["sim"] = s; print("SIM raw", s.get("raw"), "chain", s.get("g_chain_cal_ps0.8"), flush=True)
    json.dump(cv, open(os.path.join(od, "res.json"), "w"), indent=1); print("done", f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
