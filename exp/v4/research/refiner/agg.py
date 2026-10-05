"""Pair-level refiner: the per-matching rows of a config are pooled per (tile, other) candidate -- mean of every
feature over the matchings that produce the pair, plus how many matchings do (/K) and the spread of the offset --
and one model decides each candidate pair (nested 5-fold subject CV; optional inner CV for nested thresholds).
Optionally the exercise identities of `cur` and `oth` as categorical features.
  python agg.py --cfg ctx_all [--cls] [--inner] [--rounds 300]"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1
import rlib as R
import exp as E
import exp2 as E2

ALL = dict(E.CFGS, **E2.CFGS2)


def get_rows(d, so, cfg, fin, tag="oof", cache=True):
    p = os.path.join(R.HERE, "rows", f"{cfg}_{tag}.npz")
    if cache and os.path.exists(p):
        z = np.load(p); return z["X"], z["tl"], z["ot"], z["mid"]
    c = ALL[cfg]; Eu = E2.centred_unit(d["vmean_o" if so.o == "o" else "vmean_t"], d["sbj"] if so.o == "o" else d["tsbj"])
    X, tl, ot, mid = E2.build2(so, fin, c, Eu)
    if cache:
        os.makedirs(os.path.dirname(p), exist_ok=True); np.savez(p, X=X, tl=tl, ot=ot, mid=mid)
    return X, tl, ot, mid


def pool(X, tl, ot, K):
    key = tl * 32 + ot; uk, inv, cn = np.unique(key, return_inverse=True, return_counts=True)
    S = np.zeros((len(uk), X.shape[1])); np.add.at(S, inv, X)
    M = S / cn[:, None]
    off = X[:, 0]; mn = np.full(len(uk), 99.0); mx = np.full(len(uk), -99.0)
    np.minimum.at(mn, inv, off); np.maximum.at(mx, inv, off)
    Z = np.concatenate([M, (cn / K)[:, None], mn[:, None], mx[:, None]], 1).astype(np.float32)
    return Z, uk // 32, uk % 32


def pair_flips(fin, g, o, p, thr):
    keep = p > thr; g, o, p = g[keep], o[keep], p[keep]
    idx = np.lexsort((-p, g)); g, o = g[idx], o[idx]; first = np.r_[True, g[1:] != g[:-1]] if len(g) else np.zeros(0, bool)
    new = fin.copy(); new[g[first]] = o[first]; return new


def fit_pred(Z, T, tr, te, params, rounds, cat):
    import lightgbm as lgb
    ds = lgb.Dataset(Z[tr], T[tr], categorical_feature=cat if cat else "auto")
    return lgb.train(params, ds, rounds).predict(Z[te])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--cls", action="store_true")
    ap.add_argument("--inner", action="store_true"); ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--leaves", type=int, default=31); ap.add_argument("--mdl", type=int, default=50)
    ap.add_argument("--name", default="")
    a = ap.parse_args(); name = a.name or ("agg_" + a.cfg + ("_cls" if a.cls else ""))
    params = dict(E.PARAMS, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.mdl)
    d = R.load_cache(); y, fold, fin = d["y"], d["fold"], d["fin_o"]
    so = E.Side(d, "oof"); E.log("side ready")
    X, tl, ot, mid = get_rows(d, so, a.cfg, fin); K = len(so.S)
    Z, g, o = pool(X, tl, ot, K); cat = []
    if a.cls:
        Z = np.concatenate([Z, fin[g][:, None].astype(np.float32), o[:, None].astype(np.float32)], 1); cat = [Z.shape[1] - 2, Z.shape[1] - 1]
    T = ((y[g] == o) & (y[g] != fin[g])).astype(int); Fg = fold[g]
    E.log(f"{name}: rows {X.shape} -> pairs {Z.shape}, {T.mean():.3f} positive")
    pr = np.zeros(len(T)); inn = np.full((R.FOLDS, len(T)), np.nan)
    for k in range(R.FOLDS):
        pr[Fg == k] = fit_pred(Z, T, Fg != k, Fg == k, params, a.rounds, cat)
    if a.inner:
        for k in range(R.FOLDS):
            for k2 in range(R.FOLDS):
                if k2 != k:
                    inn[k, Fg == k2] = fit_pred(Z, T, (Fg != k) & (Fg != k2), Fg == k2, params, a.rounds, cat)
    for thr in (0.4, 0.45, 0.5, 0.55, 0.6):
        new = pair_flips(fin, g, o, pr, thr)
        E.log(f"  thr {thr}: F1 {macro_f1(y, new):.4f} | " + " ".join(f"{macro_f1(y[fold == f], new[fold == f]):.4f}" for f in range(R.FOLDS)))
    new = pair_flips(fin, g, o, pr, 0.5)
    msg = ""
    if a.inner:
        out = fin.copy(); picks = []
        for k in range(R.FOLDS):
            m = fold != k; best = None
            for thr in np.round(np.arange(0.3, 0.71, 0.05), 2):
                lab = pair_flips(fin, g[Fg != k], o[Fg != k], inn[k][Fg != k], thr); s = macro_f1(y[m], lab[m])
                if best is None or s > best[0]:
                    best = (s, thr)
            picks.append(best[1]); lab = pair_flips(fin, g[Fg == k], o[Fg == k], pr[Fg == k], best[1]); out[fold == k] = lab[fold == k]
        msg = f"; NESTED thr {macro_f1(y, out):.4f} picks {picks} | " + " ".join(f"{macro_f1(y[fold == f], out[fold == f]):.4f}" for f in range(R.FOLDS))
    E.log(f"RESULT {name}: F1 {macro_f1(y, fin):.4f} -> {macro_f1(y, new):.4f} | per fold "
          + " ".join(f"{macro_f1(y[fold == f], new[fold == f]):.4f}" for f in range(R.FOLDS)) + msg)
    np.savez(os.path.join(R.HERE, "preds", f"{name}.npz"), g=g, o=o, pr=pr, inner=inn.astype(np.float32), T=T)


if __name__ == "__main__":
    main()
