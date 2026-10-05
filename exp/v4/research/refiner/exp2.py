"""Refiner research, part 2: segment-context features on top of exp.py's feature blocks.
ctx: along the matching chain, the run of `cur` behind the tile and the tiles of `oth` ahead (towards the boundary),
     up to D tiles each: run lengths, cosine similarity of the tile's (per-subject centred) video embedding to the mean
     embedding of either side, and energy / video-motion differences to either side.
  python exp2.py --cfg ctx [--inner]"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1
import rlib as R
import exp as E

CFGS2 = {
    "ctx": dict(H=3, feats=(), ctx=True),
    "ctx_vote": dict(H=3, feats=("vote",), ctx=True),
    "ctx_all": dict(H=3, feats=("fit", "link", "vote"), ctx=True),
    "ctx_fit": dict(H=3, feats=("fit",), ctx=True),
    "ctx_link": dict(H=3, feats=("link",), ctx=True),
}
D = 5


def centred_unit(V, sbj):
    out = np.empty_like(V, dtype=np.float32)
    for s in np.unique(sbj):
        ii = sbj == s; X = V[ii].astype(np.float32); X -= X.mean(0); out[ii] = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    return out


def ctx_feats(side, fin, m, ix, Eu):
    su = side.S[m]; nx, pv, _ = R.chains(su, D)
    g, cur, oth, off = ix["g"], ix["cur"], ix["oth"], ix["off"]
    fwd = off <= 0                                              # boundary ahead in the successor direction
    ahead = np.stack([np.where(fwd, nx[k][g], pv[k][g]) for k in range(1, D + 1)], 1)
    behind = np.stack([np.where(fwd, pv[k][g], nx[k][g]) for k in range(1, D + 1)], 1)
    la = np.where(ahead >= 0, fin[np.maximum(ahead, 0)], -1); lb = np.where(behind >= 0, fin[np.maximum(behind, 0)], -1)
    mo = la == oth[:, None]; mc = lb == cur[:, None]
    run_c = np.cumprod(mc, 1).sum(1); n_o = mo.sum(1)
    first_o = np.where(mo.any(1), mo.argmax(1), D)             # tiles of cur (or other) before the first oth tile
    out = [run_c, n_o, first_o]
    eg = Eu[g]
    so = np.zeros_like(eg); sc_ = np.zeros_like(eg)
    for k in range(D):
        so += Eu[np.maximum(ahead[:, k], 0)] * mo[:, k:k + 1]; sc_ += Eu[np.maximum(behind[:, k], 0)] * mc[:, k:k + 1]
    cos_o = np.where(n_o > 0, (eg * so).sum(1) / (np.linalg.norm(so, axis=1) + 1e-6), 0)
    cos_c = np.where(run_c > 0, (eg * sc_).sum(1) / (np.linalg.norm(sc_, axis=1) + 1e-6), 0)
    out += [cos_o, cos_c, cos_o - cos_c]
    for v in (side.ener, side.vmot):
        vo = np.where(n_o > 0, (v[np.maximum(ahead, 0)] * mo).sum(1) / np.maximum(n_o, 1), 0)
        vc = np.where(run_c > 0, (v[np.maximum(behind, 0)] * mc).sum(1) / np.maximum(run_c, 1), 0)
        out += [np.abs(v[g] - vo) - np.abs(v[g] - vc), v[g] - vo, v[g] - vc]
    # class evidence of the two sides: mean blend margin oth-vs-cur over either side's tiles
    Bo_ = (side.B[np.maximum(ahead, 0), oth[:, None]] - side.B[np.maximum(ahead, 0), cur[:, None]])
    Bc_ = (side.B[np.maximum(behind, 0), oth[:, None]] - side.B[np.maximum(behind, 0), cur[:, None]])
    out += [np.where(n_o > 0, (Bo_ * mo).sum(1) / np.maximum(n_o, 1), 0), np.where(run_c > 0, (Bc_ * mc).sum(1) / np.maximum(run_c, 1), 0)]
    return np.stack([np.asarray(o, np.float32) for o in out], 1)


def rows_one2(side, fin, m, H, feats, ctx, Eu):
    X, g, oth = E.rows_one(side, fin, m, H, feats)
    if ctx:
        su = side.S[m]
        _, ix = R.rows(fin, su, side.B, side.lP, side.lQ, side.lw, side.sens, side.ener, side.post, side.vmot, side.vmean, H, side.dnext[m])
        assert np.array_equal(ix["g"], g)
        X = np.concatenate([X, ctx_feats(side, fin, m, ix, Eu)], 1)
    return X, g, oth


def build2(side, fin, c, Eu, members=None):
    members = range(len(side.S)) if members is None else members
    out = [rows_one2(side, fin, m, c["H"], c["feats"], c.get("ctx"), Eu) for m in members]
    X = np.concatenate([o[0] for o in out]); tl = np.concatenate([o[1] for o in out]); ot = np.concatenate([o[2] for o in out])
    mid = np.concatenate([np.full(len(o[1]), m) for m, o in zip(members, out)])
    if "vote" in c["feats"]:
        key = tl * 32 + ot; uk, inv, cn = np.unique(key, return_inverse=True, return_counts=True)
        X = np.concatenate([X, (cn[inv] / len(members)).astype(np.float32)[:, None]], 1)
    return X, tl, ot, mid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--inner", action="store_true")
    ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--name", default="")
    a = ap.parse_args(); c = CFGS2[a.cfg]; name = a.name or a.cfg
    d = R.load_cache(); y, fold, fin = d["y"], d["fold"], d["fin_o"]
    so = E.Side(d, "oof"); Eu = centred_unit(d["vmean_o"], d["sbj"]); E.log("side ready")
    X, tl, ot, mid = build2(so, fin, c, Eu); Fr = fold[tl]
    T = ((y[tl] == ot) & (y[tl] != fin[tl])).astype(int)
    E.log(f"{name}: {X.shape} rows x feats, {T.mean():.3f} should flip")
    pr, inn = E.cv(X, T, Fr, E.PARAMS, a.rounds, a.inner)
    K = len(so.S); new, nf = R.flips(fin, tl, ot, pr, K)
    E.log(f"RESULT {name}: {nf} flips, F1 {macro_f1(y, fin):.4f} -> {macro_f1(y, new):.4f} | per fold "
          + " ".join(f"{macro_f1(y[fold == f], new[fold == f]):.4f}" for f in range(R.FOLDS)))
    np.savez(os.path.join(R.HERE, "preds", f"{name}.npz"), tl=tl, ot=ot, mid=mid, pr=pr, inner=inn.astype(np.float32), T=T, K=K)


if __name__ == "__main__":
    main()
