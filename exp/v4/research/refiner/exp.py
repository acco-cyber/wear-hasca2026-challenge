"""Boundary-refiner research on the fused K7+K9 decode. For a config: build rows along the 16 matchings, train the
refiner with 5-fold subject CV (outer predictions) and, with --inner, the 4x4 inner CV used for nested choices
(threshold, config selection, second iteration). Predictions are cached in preds/<cfg>.npz.
  python exp.py --cfg base [--inner]"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1
import rlib as R

T0 = time.time()
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3)
CFGS = {
    "base": dict(H=3, feats=()),
    "h4": dict(H=4, feats=()),
    "h5": dict(H=5, feats=()),
    "fit": dict(H=3, feats=("fit",)),
    "link": dict(H=3, feats=("link",)),
    "vote": dict(H=3, feats=("vote",)),
    "fit_link_vote": dict(H=3, feats=("fit", "link", "vote")),
    "all_h4": dict(H=4, feats=("fit", "link", "vote")),
    "perfit": dict(H=3, feats=(), perfit=True),
    "kref": dict(H=3, feats=("fit", "kref")),
}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


class Side:
    """everything needed to build rows on one side (oof / test)"""
    def __init__(self, d, s):
        o = "o" if s == "oof" else "t"; self.o = o
        self.B = d[f"B{o}"]; self.lP = np.log(np.clip(d[f"P{o}"].astype(np.float64), 1e-9, None)).astype(np.float32)
        self.lQ = np.log(d[f"Q{o}"].astype(np.float64) + 1e-9).astype(np.float32); self.lw = d[f"lw{o}"]
        self.sens, self.ener, self.post, self.vmot, self.vmean = d[f"sens_{o}"], d[f"ener_{o}"], d[f"post_{o}"], d[f"vmot_{o}"], d[f"vmean_{o}"]
        self.S, self.SC = d[f"L{o}_s"], d[f"L{o}_c"]
        self.fit = []
        for i in range(2):
            self.fit.append(dict(lP=np.log(np.clip(d[f"f{i}_P{o}"].astype(np.float64), 1e-9, None)).astype(np.float32),
                                 lQ=np.log(d[f"f{i}_Q{o}"].astype(np.float64) + 1e-9).astype(np.float32), B=d[f"f{i}_B{o}"],
                                 lab=d[f"f{i}_Q{o}"].argmax(1), kref=d[f"f{i}_ref{o}"].astype(np.int64)))
        self.dnext = [R.vnorm_next(self.vmean, su) for su in self.S]
        n = self.S.shape[1]; self.P = np.full_like(self.S, -1)
        for m, su in enumerate(self.S):
            k = su >= 0; self.P[m, su[k]] = np.flatnonzero(k)


def rows_one(side, fin, m, H, feats):
    su = side.S[m]
    X, ix = R.rows(fin, su, side.B, side.lP, side.lQ, side.lw, side.sens, side.ener, side.post, side.vmot, side.vmean, H, side.dnext[m])
    g, cur, oth = ix["g"], ix["cur"], ix["oth"]; ext = []
    if "fit" in feats or "kref" in feats:
        for f in side.fit:
            ext += [R.pair(f["lP"], g, cur, oth), R.pair(f["lQ"], g, cur, oth), R.pair(f["B"], g, cur, oth),
                    (f["lab"][g] == cur), (f["lab"][g] == oth)]
            if "kref" in feats:
                ext += [(f["kref"][g] == oth), (f["kref"][g] == cur)]
        ext += [(side.fit[0]["lab"][g] == side.fit[1]["lab"][g])]
    if "link" in feats:
        sc = side.SC[m]; K = len(side.S)
        stab_n = (side.S[:, g] == su[g][None]).mean(0) * (su[g] >= 0)
        prv = ix["prv"]; stab_p = (side.P[:, g] == prv[None]).mean(0) * (prv >= 0)
        stab_b = (side.S[:, ix["i"]] == ix["j"][None]).mean(0)
        ext += [np.where(su[g] >= 0, sc[g], -9.0), R.take(sc, prv, -9.0), sc[ix["i"]], stab_n, stab_p, stab_b]
    if "vote" in feats:
        NL = np.where(side.S[:, g] >= 0, fin[np.maximum(side.S[:, g], 0)], -1)
        PL = np.where(side.P[:, g] >= 0, fin[np.maximum(side.P[:, g], 0)], -1)
        ext += [(NL == oth[None]).mean(0), (PL == oth[None]).mean(0), (NL == cur[None]).mean(0), (PL == cur[None]).mean(0)]
    if ext:
        X = np.concatenate([X, np.stack([np.asarray(e, np.float32) for e in ext], 1)], 1)
    return X, g, oth


def build(side, fin, H, feats, members=None):
    members = range(len(side.S)) if members is None else members
    out = [rows_one(side, fin, m, H, feats) for m in members]
    X = np.concatenate([o[0] for o in out]); tl = np.concatenate([o[1] for o in out]); ot = np.concatenate([o[2] for o in out])
    mid = np.concatenate([np.full(len(o[1]), m) for m, o in zip(members, out)])
    if "vote" in feats:                                   # rows with the same (tile, other) over all matchings / K
        key = tl * 32 + ot; uk, inv, cn = np.unique(key, return_inverse=True, return_counts=True)
        X = np.concatenate([X, (cn[inv] / len(members)).astype(np.float32)[:, None]], 1)
    return X, tl, ot, mid


def train_pred(X, T, tr, te, params, rounds):
    import lightgbm as lgb
    return lgb.train(params, lgb.Dataset(X[tr], T[tr]), rounds).predict(X[te])


def cv(X, T, Fr, params, rounds, inner):
    """outer: pr[i] from a model without row i's fold; inner[k]: for rows of folds != k, predictions from models
    trained without fold k AND without the row's own fold"""
    pr = np.zeros(len(T))
    for k in range(R.FOLDS):
        pr[Fr == k] = train_pred(X, T, Fr != k, Fr == k, params, rounds)
    inn = np.full((R.FOLDS, len(T)), np.nan)
    if inner:
        for k in range(R.FOLDS):
            for k2 in range(R.FOLDS):
                if k2 != k:
                    inn[k, Fr == k2] = train_pred(X, T, (Fr != k) & (Fr != k2), Fr == k2, params, rounds)
            log(f"  inner fold {k} done")
    return pr, inn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--inner", action="store_true")
    ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=31); ap.add_argument("--mdl", type=int, default=50)
    ap.add_argument("--name", default="")
    a = ap.parse_args(); c = CFGS[a.cfg]; name = a.name or a.cfg
    params = dict(PARAMS, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.mdl)
    d = R.load_cache(); y, fold, fin = d["y"], d["fold"], d["fin_o"]
    so = Side(d, "oof"); log("side ready")
    X, tl, ot, mid = build(so, fin, c["H"], c["feats"]); Fr = fold[tl]
    T = ((y[tl] == ot) & (y[tl] != fin[tl])).astype(int)
    log(f"{name}: {X.shape} rows x feats, {T.mean():.3f} should flip")
    if c.get("perfit"):
        pr = np.zeros(len(T)); inn = np.full((R.FOLDS, len(T)), np.nan)
        for grp in (mid < 8, mid >= 8):
            p_, i_ = cv(X[grp], T[grp], Fr[grp], params, a.rounds, a.inner); pr[grp] = p_; inn[:, grp] = i_
    else:
        pr, inn = cv(X, T, Fr, params, a.rounds, a.inner)
    K = len(so.S)
    new, nf = R.flips(fin, tl, ot, pr, K)
    log(f"RESULT {name}: {nf} flips, F1 {macro_f1(y, fin):.4f} -> {macro_f1(y, new):.4f} | per fold "
        + " ".join(f"{macro_f1(y[fold == f], new[fold == f]):.4f}" for f in range(R.FOLDS)))
    os.makedirs(os.path.join(R.HERE, "preds"), exist_ok=True)
    np.savez(os.path.join(R.HERE, "preds", f"{name}.npz"), tl=tl, ot=ot, mid=mid, pr=pr, inner=inn.astype(np.float32), T=T, K=K)


if __name__ == "__main__":
    main()
