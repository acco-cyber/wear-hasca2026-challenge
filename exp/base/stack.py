"""Level-2 stacker on per-(second,limb) OOF log-probs of several base models, subject-disjoint CV (same 5 folds).
python stack.py --models work/lgbm_v1,work/fusion_v1 --out stack_lr_v1f [--kind lr|lgb] [--C 0.1] [--fam]
Model paths: a directory containing oof.npy (N,4,19) and test.npy (12234,19)."""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.linear_model import LogisticRegression

def feats(Ls, limb_oh, fam=False):
    """Ls: list of (n,19) log-probs."""
    parts = []
    for L in Ls:
        parts.append(L)
        if fam:
            P = np.exp(L); F = np.stack([P[:, FAMILY == f].sum(1) for f in range(8)], 1); parts.append(np.log(np.clip(F, 1e-6, 1)))
    parts.append(limb_oh)
    return np.concatenate(parts, 1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="lr"); ap.add_argument("--C", type=float, default=0.05); ap.add_argument("--fam", action="store_true")
    ap.add_argument("--sub", type=int, default=2, help="training rows: this many random valid limbs per second")
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    paths = [p if os.path.isabs(p) else os.path.join(r"E:\Claude code\wear", p) for p in a.models.split(",")]
    oofs = [np.load(os.path.join(p, "oof.npy")) for p in paths]; tests = [np.load(os.path.join(p, "test.npy")) for p in paths]
    for p, o in zip(paths, oofs): eval_single(o, name=os.path.basename(p))
    m = meta(); y = m.y.to_numpy(); N = len(m); fold = sec_fold()
    valid = ~np.isnan(oofs[0][:, :, 0])
    for o in oofs[1:]: valid &= ~np.isnan(o[:, :, 0])
    sec, limb = np.where(valid)
    Ls = [to_log(o[sec, limb]) for o in oofs]
    X = feats(Ls, np.eye(4, dtype=np.float32)[limb], a.fam); yy = y[sec]; ff = fold[sec]
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv")); tl = np.array([LIMBS.index(l) for l in tm.sensor_location])
    Xte = feats([to_log(t) for t in tests], np.eye(4, dtype=np.float32)[tl], a.fam)
    # training-row thinning (random limbs per second)
    rng = np.random.RandomState(3); keep = np.zeros(len(sec), bool)
    order = rng.permutation(len(sec)); cnt = {}
    r = rng.rand(len(sec));
    # keep a.sub random valid limbs per second: rank limbs per second by random key
    key = sec.astype(np.float64) + r * 0.5
    srt = np.argsort(key, kind="stable"); rank = np.zeros(len(sec), int)
    s_sorted = sec[srt]; start = np.r_[0, np.where(np.diff(s_sorted) != 0)[0] + 1]
    pos = np.arange(len(sec)) - np.repeat(start, np.diff(np.r_[start, len(sec)]))
    rank[srt] = pos; keep = rank < a.sub
    print("rows", len(sec), "train rows", keep.sum(), "feats", X.shape[1], flush=True)
    def fit(Xt, yt):
        if a.kind == "lr":
            mdl = LogisticRegression(C=a.C, max_iter=300, tol=1e-4)
            mdl.fit(Xt, yt); return mdl
        import lightgbm as lgb
        params = dict(objective="multiclass", num_class=NC, learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.5,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=63, num_threads=4, verbose=-1, seed=0)
        return lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=200)
    def pred(mdl, Xp):
        return mdl.predict_proba(Xp) if a.kind == "lr" else mdl.predict(Xp)
    oof = np.full((N, 4, NC), np.nan, np.float32)
    for k in range(5):
        tr = keep & (ff != k); va = ff == k
        mdl = fit(X[tr], yy[tr]); p = pred(mdl, X[va]); oof[sec[va], limb[va]] = p
        print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)
    f = eval_single(oof, name=a.out)
    mdl = fit(X[keep], yy[keep]); test = pred(mdl, Xte).astype(np.float32)
    od = os.path.join(EXP, a.out); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), oof); np.save(os.path.join(od, "test.npy"), test)
    res = dict(models=a.models, kind=a.kind, C=a.C, fam=a.fam, oof_f1=f)
    if a.sim:
        s, _ = run_sim(os.path.join(od, "oof.npy"), a.out); res["sim"] = s; print("SIM", s.get("raw"), s.get("g_chain_cal_ps0.8"), flush=True)
    json.dump(res, open(os.path.join(od, "res.json"), "w"), indent=1); print(res, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
