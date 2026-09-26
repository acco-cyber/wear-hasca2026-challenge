"""Train / validate the UEC video members (port of train_video_mlp.py / train_video_cnn.py + run_subject_cv.py).

--protocol manifest : run_subject_cv --num-folds 5 --exclude-file-id-suffix-2 (contiguous subject folds over 0..21),
                      early stopping + plateau on the held-out fold (exactly as the repo), OOF over concatenated folds,
                      fold-model test predictions averaged (= repo's cv_test_probabilities).
--protocol standard : our 5-fold perm split (RandomState(0)), fixed epochs (no peeking), eval on 1-s tiles
                      with one random valid limb (seed 5), purity >= 0.8 (exp/base/common.eval_single).
--protocol full     : all non-_2 train windows, fixed epochs, test probabilities in test-id order.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vport import *

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["pca", "raw"], required=True)
ap.add_argument("--member", choices=["a", "b", "cnn"], required=True)
ap.add_argument("--protocol", choices=["manifest", "standard", "full"], required=True)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--folds", type=int, nargs="*", default=None)
ap.add_argument("--seeds", type=int, nargs="*", default=[42])
ap.add_argument("--tag", default="")
args = ap.parse_args()
fd = os.path.join(FEAT, args.backend)
rd = os.path.join(OUT, "runs", f"{args.backend}_{args.member}_{args.protocol}{args.tag}"); os.makedirs(rd, exist_ok=True)
logf = open(os.path.join(rd, "log.txt"), "a")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()
t0 = time.time()

# ---------------------------------------------------------------- load features
def load_train():
    if args.backend == "pca":
        meta = pd.read_csv(os.path.join(fd, "meta.csv"))
        ld = lambda k: np.load(os.path.join(fd, f"{k}.npy"), mmap_mode="r")
        get = {k: ld(k) for k in (["cnn"] if args.member == "cnn" else [f"{args.member}_agg", f"{args.member}_sc"])}
        return meta, get
    sd = os.path.join(fd, "sessions")
    sess = sorted(f[:-9] for f in os.listdir(sd) if f.endswith("_meta.csv"))
    metas, parts = [], {}
    keys = ["cnn"] if args.member == "cnn" else [f"{args.member}_agg", f"{args.member}_sc"]
    for s in sess:
        metas.append(pd.read_csv(os.path.join(sd, f"{s}_meta.csv")))
        for k in keys:
            parts.setdefault(k, []).append(np.load(os.path.join(sd, f"{s}_{k}.npy")))
    return pd.concat(metas, ignore_index=True), {k: np.concatenate(v) for k, v in parts.items()}

meta, F = load_train()
TF = {k: np.load(os.path.join(fd, f"test_{k}.npy"), mmap_mode="r") for k in F}
is2 = meta.session.str.count("_").to_numpy() >= 2              # sbj_0_2, sbj_14_2
trainable = (~is2) & (meta.y.to_numpy() >= 0) & (meta.pur.to_numpy() >= 0.8) & (meta.ra_ok.to_numpy() == 1)
y_all = meta.y.to_numpy().astype(np.int64)
sbj_all = meta.sbj.to_numpy()
log(f"=== {args} rows={len(meta)} trainable={trainable.sum()} sessions={meta.session.nunique()} load {time.time()-t0:.0f}s")


def build_X(fit_rows):
    """member input matrix; scalar block z-scored with statistics of fit_rows (float16 storage)."""
    if args.member == "cnn":
        return np.asarray(F["cnn"]), np.asarray(TF["cnn"])
    agg, sc = F[f"{args.member}_agg"], np.asarray(F[f"{args.member}_sc"], np.float32)
    mu = sc[fit_rows].mean(0); sd = np.maximum(sc[fit_rows].std(0), 1e-6)
    X = np.empty((len(sc), agg.shape[1] + sc.shape[1]), np.float16)
    X[:, :agg.shape[1]] = agg; X[:, agg.shape[1]:] = (sc - mu) / sd
    tagg, tsc = TF[f"{args.member}_agg"], np.asarray(TF[f"{args.member}_sc"], np.float32)
    XT = np.empty((len(tsc), X.shape[1]), np.float16)
    XT[:, :agg.shape[1]] = tagg; XT[:, agg.shape[1]:] = (tsc - mu) / sd
    return X, XT

kind = "cnn" if args.member == "cnn" else "mlp"
torch = get_torch()

# ---------------------------------------------------------------- protocols
if args.protocol == "manifest":
    subjects = sorted(np.unique(sbj_all[~is2]).tolist())
    base, rem = divmod(len(subjects), 5)
    folds, st = [], 0
    for i in range(5):
        sz = base + (1 if i < rem else 0); folds.append(subjects[st:st + sz]); st += sz
    oof = np.full((len(meta), NC), np.nan, np.float32)
    tests, rows = [], []
    for fi, vs in enumerate(folds):
        if args.folds is not None and fi not in args.folds:
            continue
        tr = trainable & ~np.isin(sbj_all, vs); va = trainable & np.isin(sbj_all, vs)
        X, XT = build_X(np.where(tr)[0])
        itr, iva = np.where(tr)[0], np.where(va)[0]
        log(f"[fold {fi}] val_subjects={vs} train={len(itr)} val={len(iva)}")
        model, info = train_member(kind, X[itr], y_all[itr], X[iva], y_all[iva], log=log)
        pv = predict(model, X[iva], torch); oof[iva] = pv
        pt = predict(model, XT, torch); tests.append(pt)
        np.save(os.path.join(rd, f"test_fold{fi}.npy"), pt)
        rows.append({"fold": fi, "val_subjects": vs, "best_epoch": info["best_epoch"], "val_macro_f1": macro_f1(y_all[iva], pv.argmax(1))})
        log(f"[fold {fi}] best_epoch={info['best_epoch']} val_macro_f1={rows[-1]['val_macro_f1']:.4f} {time.time()-t0:.0f}s")
        with open(os.path.join(rd, f"hist_fold{fi}.json"), "w") as f: json.dump(info["history"], f)
    np.save(os.path.join(rd, "oof.npy"), oof)
    done = ~np.isnan(oof[:, 0]) & trainable
    res = {"folds": rows, "oof_macro_f1": macro_f1(y_all[done], oof[done].argmax(1)) if done.any() else None,
           "median_best_epoch": int(np.median([r["best_epoch"] for r in rows])) if rows else None}
    if len(tests) == 5:
        np.save(os.path.join(rd, "test_cvavg.npy"), np.mean(tests, 0).astype(np.float32))
    json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)
    log("RESULT " + json.dumps(res))

elif args.protocol == "standard":
    assert args.epochs, "--epochs required (median best epoch of the manifest CV)"
    tile_rows = meta.tile.to_numpy() >= 0
    tm = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    fo = {s: i % 5 for i, s in enumerate(np.random.RandomState(0).permutation(np.unique(tm.sbj)))}
    fold_all = np.array([fo[s] for s in sbj_all])
    # map tiles -> train_meta rows
    key = pd.DataFrame({"session": meta.session, "t": meta.tile, "r": np.arange(len(meta))})[tile_rows]
    tmk = tm[["session", "t"]].reset_index().merge(key, on=["session", "t"], how="left")
    P = np.full((len(tm), NC), np.nan, np.float32)
    for fi in range(5):
        if args.folds is not None and fi not in args.folds:
            continue
        tr = trainable & (fold_all != fi)
        X, _ = build_X(np.where(tr)[0])
        itr = np.where(tr)[0]
        ps = []
        for sd_ in args.seeds:
            model, info = train_member(kind, X[itr], y_all[itr], fixed_epochs=args.epochs, seed=sd_, log=log)
            te = np.where(tile_rows & (fold_all == fi))[0]
            ps.append(predict(model, X[te], torch))
        pr = np.mean(ps, 0)
        r2t = pd.Series(np.arange(len(tm)), index=None)
        sel = tmk.r.notna().to_numpy()
        m = dict(zip(te, range(len(te))))
        for trow, r in zip(tmk.index[sel], tmk.r[sel].astype(int)):
            if r in m: P[trow] = pr[m[r]]
        log(f"[std fold {fi}] train={len(itr)} pred_tiles={len(te)} {time.time()-t0:.0f}s")
    np.save(os.path.join(rd, "oof_tiles.npy"), P)
    sys.path.insert(0, r"E:\Claude code\wear\exp\base")
    import common
    oof4 = np.repeat(P[:, None, :], 4, axis=1)
    f = common.eval_single(oof4, name=f"uec_video_{args.member}_{args.backend}")
    res = {"std_macro_f1": f, "epochs": args.epochs, "missing_tiles": int(np.isnan(P[:, 0]).sum())}
    json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)
    log("RESULT " + json.dumps(res))

else:  # full
    assert args.epochs, "--epochs required"
    tr = trainable
    X, XT = build_X(np.where(tr)[0])
    itr = np.where(tr)[0]
    ps = []
    for sd_ in args.seeds:
        model, info = train_member(kind, X[itr], y_all[itr], fixed_epochs=args.epochs, seed=sd_, log=log)
        ps.append(predict(model, XT, torch))
        torch.save(model.state_dict(), os.path.join(rd, f"model_seed{sd_}.pt"))
    pt = np.mean(ps, 0).astype(np.float32)
    np.save(os.path.join(rd, "test.npy"), pt)
    res = {"n_train": int(len(itr)), "epochs": args.epochs, "seeds": args.seeds, "test_shape": list(pt.shape),
           "test_pred_dist": np.bincount(pt.argmax(1), minlength=NC).tolist()}
    json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)
    log("RESULT " + json.dumps(res))
log(f"DONE {time.time()-t0:.0f}s")
