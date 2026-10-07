"""(b) our one-random-limb tiles + simulated 2025 per-limb chains: re-score the L3 link candidates with chain evidence.
For a candidate A->B (A at second t in limb LA, B hypothesised at t+1 in limb LB) walk chain LA around A and chain LB
around B; under the hypothesis the walks are two tracks of the same timeline, and our tiles occupy exactly ONE limb per
second, so positions where both tracks carry one of our tiles ("conflicts") refute the link. Same-limb pairs: is B the
chain successor of A. Bridges: if the first own tile after A on chain LA is 2 steps ahead (C), the L3 log-odds of B->C;
likewise D->A backwards. A nested (subject-fold) LightGBM combines L3 log-odds + these features; kernel lk_assign gives
the matching.  python build_links.py --chain {oracle,nll3,lgb,none} [--anchor 1.0] [--W 16] [--tag T] [--save]"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
from common import *
sys.path.insert(0, os.path.join(W, "exp", "v4"))
from relink import lk_assign, perturbed, qnorm

ap = argparse.ArgumentParser(); ap.add_argument("--chain", default="oracle"); ap.add_argument("--anchor", type=float, default=1.0)
ap.add_argument("--W", type=int, default=16); ap.add_argument("--tag", default=""); ap.add_argument("--save", action="store_true")
ap.add_argument("--fits", default="K7"); ap.add_argument("--members", type=int, default=8); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--chain_err", type=float, default=0.0, help="oracle chains with this fraction of links rewired at random (sensitivity)")
ap.add_argument("--topk", type=int, default=50, help="L3 candidates kept per row (0 = all)")
ap.add_argument("--stage2", action="store_true", help="second pass with gap-consistency features from the stage-1 matching")
ap.add_argument("--iters", type=int, default=1, help="number of gap-feature passes (each uses the previous pass's matching)")
a = ap.parse_args()
T0 = time.time()
S = load_stage(); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]; fold = S["oof_fold"]; N = len(sbj)
LZ_PATHS = [os.path.join({"K7": K7, "K9": K9}[f], "link_logodds.npz") for f in a.fits.split(",")]
LLR = {(1, 0): np.log(4 / 3), (0, 1): np.log(4 / 3), (0, 0): np.log(8 / 9)}


def chain_for(s, rows, n):
    """per limb successor / predecessor / confidence over positions (positions = seconds in rec order)"""
    rec = S["oof_rec"][rows]; nxt = np.where(np.r_[rec[1:] == rec[:-1], False], np.arange(1, n + 1), -1)
    out = []
    if a.chain in ("oracle", "none"):
        rng = np.random.default_rng(1000 + s)
        for L in range(4):
            su = nxt.copy()
            if a.chain_err > 0:                                         # rewire a random share of links (then re-fix 1:1 by swapping)
                k = np.flatnonzero((rng.random(n) < a.chain_err) & (su >= 0))
                su[k] = su[rng.permutation(k)]
            out.append((su, np.ones(n, np.float32)))
    else:
        z = np.load(os.path.join(OUT, "cache", f"chain_{a.chain}_s{s}.npz")); assert (z["rows"] == rows).all()
        for L in range(4):
            su = z[f"succ{L}"].astype(np.int64)
            cf = z[f"conf{L}"].astype(np.float32) if f"conf{L}" in z.files else np.ones(n, np.float32)
            out.append((su, cf))
    res = []
    for su, cf in out:
        pr = np.full(n, -1, np.int64); pr[su[su >= 0]] = np.flatnonzero(su >= 0)
        cb = np.zeros(n, np.float32); cb[su[su >= 0]] = cf[su >= 0]                 # confidence of the link INTO each node
        res.append((su, pr, cf, cb))
    return res


def walks(su, cf, n, Wn):
    """node reached after k=1..W steps along su and the cumulative confidence"""
    F = np.full((n, Wn), -1, np.int64); C = np.zeros((n, Wn), np.float32); cur = np.arange(n); c = np.ones(n, np.float32)
    for k in range(Wn):
        ok = cur >= 0; nx = np.full(n, -1, np.int64); nx[ok] = su[cur[ok]]; c = np.where(nx >= 0, c * np.where(ok, cf[np.maximum(cur, 0)], 0), 0)
        F[:, k] = nx; C[:, k] = c; cur = nx
    return F, C


def subject_table(s):
    t0 = time.time()
    loc = np.flatnonzero(sbj == s); n = len(loc)                          # link-local order
    rows = loc[np.lexsort((S["oof_start"][loc], S["oof_rec"][loc]))]      # positions (seconds)
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    pos = p_of[loc]                                                       # link-local index -> position
    owner = l_of[rows]                                                    # position -> link-local index
    lim = sens[loc]                                                       # limb of each our tile (link-local)
    rng = np.random.default_rng(s); anch = rng.random(n) < a.anchor       # tiles whose 2025 twin is identified
    mark = np.zeros((4, n), np.int8)
    for L in range(4):
        mark[L, pos[(lim == L) & anch]] = 1
    tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
    # L3 candidates (fused if several fits)
    LZ = [np.load(p) for p in LZ_PATHS]
    if len(LZ) == 1:
        cand, Lo = LZ[0][f"oof_{s}_cand"].astype(np.int64), LZ[0][f"oof_{s}_L"].astype(np.float32)
    else:
        from relink import fuse_subject
        cand, Lo = fuse_subject([z[f"oof_{s}_cand"].astype(np.int64) for z in LZ], [z[f"oof_{s}_L"] for z in LZ], np.ones(len(LZ)) / len(LZ))
    if a.topk and cand.shape[1] > a.topk:                                 # keep the top-k L3 candidates of every row
        o = np.argsort(-np.where((cand >= 0) & (Lo > -49), Lo, -1e9), 1)[:, :a.topk]
        cand, Lo = np.take_along_axis(cand, o, 1), np.take_along_axis(Lo, o, 1)
    ch =chain_for(s, rows, n) if a.chain != "none" else None
    Wn = a.W
    if ch is not None:
        FW = [walks(c[0], c[2], n, Wn) for c in ch]; BW = [walks(c[1], c[3], n, Wn) for c in ch]
        # chain-suggested same-limb links from anchored tiles whose chain successor is an anchored own tile
        extra = []
        for L in range(4):
            for i in np.flatnonzero((lim == L) & anch):
                q = ch[L][0][pos[i]]
                if q >= 0 and mark[L, q]:
                    extra.append((i, owner[q]))
        if extra:
            ex = np.array(extra); have = (cand[ex[:, 0]] == ex[:, 1][:, None]).any(1); ex = ex[~have]
            if len(ex):
                rmin = np.where(cand >= 0, Lo, np.inf).min(1); add = np.full((n, 1), -1, np.int64); addL = np.full((n, 1), -50.0, np.float32)
                cnt = np.zeros(n, int)
                for i, j in ex:
                    if cnt[i] == 0:
                        add[i, 0] = j; addL[i, 0] = rmin[i] - 1.0; cnt[i] = 1
                cand = np.concatenate([cand, add], 1); Lo = np.concatenate([Lo, addL], 1)
    valid = (cand >= 0) & (Lo > -49)
    pi, ci = np.nonzero(valid); pj = cand[pi, ci]; lo = Lo[pi, ci]
    # L3 ranks
    rk_row = (np.argsort(np.argsort(-np.where(valid, Lo, -np.inf), 1), 1))[pi, ci]
    key = pi * n + pj; order = np.argsort(key); skey = key[order]; slo = lo[order]

    def l3(x, y_):
        k = x * n + y_; ix = np.clip(np.searchsorted(skey, k), 0, len(skey) - 1); hit = (skey[ix] == k) & (x >= 0) & (y_ >= 0)
        return np.where(hit, slo[ix], np.nan)
    colmax = np.full(n, -np.inf); np.maximum.at(colmax, pj, lo)
    rowmax = np.where(valid, Lo, -np.inf).max(1)
    f = {"lo": lo, "rk_row": rk_row.astype(np.float32), "d_rowmax": lo - rowmax[pi], "d_colmax": lo - colmax[pj],
         "same_limb": (lim[pi] == lim[pj]).astype(np.float32)}
    if ch is not None:
        LA, LB = lim[pi], lim[pj]; pA, pB = pos[pi], pos[pj]; aA, aB = anch[pi], anch[pj]
        na = len(pi)
        # tracks: relative seconds d=1..W (forward) and d=0..-(W-1) (backward), A-track on chain LA, B-track on chain LB
        mA_f = np.zeros((na, Wn), np.int8); cA_f = np.zeros((na, Wn), np.float32); mB_f = np.zeros((na, Wn), np.int8); cB_f = np.zeros((na, Wn), np.float32)
        mA_b = np.zeros((na, Wn), np.int8); cA_b = np.zeros((na, Wn), np.float32); mB_b = np.zeros((na, Wn), np.int8); cB_b = np.zeros((na, Wn), np.float32)
        nodeA1 = np.full(na, -1); nodeB0 = np.full(na, -1); dirsame = np.zeros(na, np.float32); cf1 = np.zeros(na, np.float32)
        for L in range(4):
            mA = LA == L
            Fn, Fc = FW[L][0][pA[mA]], FW[L][1][pA[mA]]; Bn, Bc = BW[L][0][pA[mA]], BW[L][1][pA[mA]]
            mA_f[mA] = np.where(Fn >= 0, mark[L][np.maximum(Fn, 0)], 0); cA_f[mA] = Fc                    # d = 1..W
            mA_b[mA, 0] = 1; cA_b[mA, 0] = 1.0                                                          # d = 0 is A
            mA_b[mA, 1:] = np.where(Bn[:, :-1] >= 0, mark[L][np.maximum(Bn[:, :-1], 0)], 0); cA_b[mA, 1:] = Bc[:, :-1]
            nodeA1[mA] = Fn[:, 0]; cf1[mA] = Fc[:, 0]
            mB = LB == L
            Fn, Fc = FW[L][0][pB[mB]], FW[L][1][pB[mB]]; Bn, Bc = BW[L][0][pB[mB]], BW[L][1][pB[mB]]
            mB_f[mB, 0] = 1; cB_f[mB, 0] = 1.0                                                          # d = 1 is B
            mB_f[mB, 1:] = np.where(Fn[:, :-1] >= 0, mark[L][np.maximum(Fn[:, :-1], 0)], 0); cB_f[mB, 1:] = Fc[:, :-1]
            mB_b[mB] = np.where(Bn >= 0, mark[L][np.maximum(Bn, 0)], 0); cB_b[mB] = Bc                     # d = 0..-(W-1)
            nodeB0[mB] = Bn[:, 0]
        same = LA == LB
        dirsame = (same & (nodeA1 == pB)).astype(np.float32)
        okA = aA.astype(np.float32)[:, None]; okB = aB.astype(np.float32)[:, None]
        mAll = np.concatenate([mA_b[:, ::-1], mA_f], 1).astype(np.float32); mBll = np.concatenate([mB_b[:, ::-1], mB_f], 1).astype(np.float32)
        cAll = np.concatenate([cA_b[:, ::-1], cA_f], 1) * okA; cBll = np.concatenate([cB_b[:, ::-1], cB_f], 1) * okB
        w = cAll * cBll * (~same)[:, None]                                  # cross-limb only
        conf = mAll * mBll
        llr = np.where(conf > 0, -3.0, np.where(mAll + mBll > 0, LLR[(1, 0)], LLR[(0, 0)]))
        ctr = Wn                                                            # column index of d = 1 (d = 0 at Wn-1)
        for h in (1, 2, 4, 8, 16):
            if h > Wn:
                continue
            sl = slice(ctr - h, ctr + h)
            f[f"conf_w{h}"] = (conf[:, sl] * w[:, sl]).sum(1); f[f"llr_w{h}"] = (llr[:, sl] * w[:, sl]).sum(1); f[f"wsum_w{h}"] = w[:, sl].sum(1)
        f["dirsame"] = np.where(same & aA, dirsame, np.nan).astype(np.float32)
        f["cf1"] = np.where(aA, cf1, np.nan).astype(np.float32)
        # same-limb: chain successor of A is an own anchored tile other than B
        nA1m = np.where(nodeA1 >= 0, mark[LA, np.maximum(nodeA1, 0)], 0)
        f["succ_is_other_own"] = np.where(aA, (nA1m == 1) & (nodeA1 != pB), np.nan).astype(np.float32)
        # bridges: first own tile ahead of A on chain LA (distance dA), first own tile behind B on chain LB (distance dB)
        dA = np.where(mA_f.any(1), mA_f.argmax(1) + 1, 99); dB = np.where(mB_b.any(1), mB_b.argmax(1) + 1, 99)   # mB_b[:,0] is d=0 -> 1 step back
        nodeC = np.full(na, -1); nodeD = np.full(na, -1)
        for L in range(4):
            mA = (LA == L) & (dA <= Wn); nodeC[mA] = FW[L][0][pA[mA], dA[mA] - 1]
            mB = (LB == L) & (dB <= Wn); nodeD[mB] = BW[L][0][pB[mB], dB[mB] - 1]
        C_ = np.where(nodeC >= 0, owner[np.maximum(nodeC, 0)], -1); D_ = np.where(nodeD >= 0, owner[np.maximum(nodeD, 0)], -1)
        f["dA"] = np.where(aA & ~same, dA, np.nan).astype(np.float32); f["dB"] = np.where(aB & ~same, dB, np.nan).astype(np.float32)
        f["bridge_BC"] = np.where(aA & ~same & (dA == 2), l3(pj, C_), np.nan).astype(np.float32)
        f["bridge_DA"] = np.where(aB & ~same & (dB == 2), l3(D_, pi), np.nan).astype(np.float32)
        f["anch_A"] = aA.astype(np.float32); f["anch_B"] = aB.astype(np.float32)
    names = list(f.keys()); X = np.stack([np.asarray(f[k], np.float32) for k in names], 1)
    yv = (tl[pi] == pj).astype(np.int8)
    extra_out = {}
    if ch is not None:                                                    # for the stage-2 gap features
        dfirst = np.full((4, n), 99, np.int64); nfirst = np.full((4, n), -1, np.int64); cfirst = np.zeros((4, n), np.float32)
        for L in range(4):
            Fn, Fc = FW[L]; mk = np.where(Fn >= 0, mark[L][np.maximum(Fn, 0)], 0); has = mk.any(1); k = mk.argmax(1)
            dfirst[L, has] = k[has] + 1; nfirst[L, has] = Fn[has, k[has]]; cfirst[L, has] = Fc[has, k[has]]
        extra_out = dict(dfirst=dfirst, nfirst=nfirst, cfirst=cfirst, pos=pos, owner=owner)
    return dict(s=s, loc=loc, pi=pi, pj=pj, X=X, y=yv, names=names, tl=tl, n=n, lim=lim, anch=anch, t=time.time() - t0, **extra_out)


subs = [int(x) for x in np.unique(sbj)]
subs_sorted = sorted(subs, key=lambda s: -(sbj == s).sum())
tabs = {t["s"]: t for t in Parallel(n_jobs=a.jobs)(delayed(subject_table)(s) for s in subs_sorted)}
names = tabs[subs[0]]["names"]
print(f"[{time.time() - T0:.0f}s] tables built; features: {names}", flush=True)
fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
X = np.concatenate([tabs[s]["X"] for s in subs]); y = np.concatenate([tabs[s]["y"] for s in subs])
fo = np.concatenate([np.full(len(tabs[s]["y"]), fold_of[s]) for s in subs])
print(f"{len(y)} candidate pairs, positives {y.sum()} ({y.mean():.4f})", flush=True)
P = dict(objective="binary", learning_rate=0.08, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3)
rk = X[:, names.index("rk_row")]; LO = X[:, names.index("lo")].astype(np.float64)
rng = np.random.default_rng(0); keep_all = (y == 1) | (rk < 10); samp = rng.random(len(y)) < 0.2
trm = keep_all | samp; wts = np.where(keep_all, 1.0, 5.0)               # subsampled easy negatives re-weighted (unbiased)
DROP = {"rk_row", "d_rowmax", "d_colmax"}                                 # competition features distort the Hungarian sum


def nested(X, names, label, base=None):
    """boosting FROM the L3 log-odds (init_score): the model learns a correction, so with no extra evidence the matching
    stays the kernel's"""
    base = LO if base is None else base.astype(np.float64)
    cols = [k for k, nm in enumerate(names) if nm not in DROP]; Xm = X[:, cols]; nm_ = [names[k] for k in cols]
    pred = np.zeros(len(y), np.float32)
    for f_ in range(5):
        tr = (fo != f_) & trm; te = fo == f_
        m = lgb.train(P, lgb.Dataset(Xm[tr], y[tr], weight=wts[tr], init_score=base[tr]), 300)
        pred[te] = m.predict(Xm[te], raw_score=True) + base[te]
        if f_ == 0:
            imp = m.feature_importance("gain"); print(f"{label} top features:", [(nm_[k], round(float(imp[k] / imp.sum()), 3)) for k in np.argsort(-imp)[:14]], flush=True)
    print(f"[{time.time() - T0:.0f}s] {label} nested scorer done", flush=True)
    return pred


pred = nested(X, names, "stage1")


def match(s, pr, members, ref):
    t = tabs[s]; n = t["n"]; pi, pj = t["pi"], t["pj"]
    cnt = np.bincount(pi, minlength=n); K = int(cnt.max()); order = np.argsort(pi, kind="stable")
    start = np.r_[0, np.cumsum(cnt)][:-1]; slot = np.arange(len(pi)) - start[pi[order]]
    cand = np.full((n, K), -1, np.int64); Lm = np.full((n, K), -50.0, np.float32)
    cand[pi[order], slot] = pj[order]; Lm[pi[order], slot] = np.clip(pr[order], -45, 45)
    res = [lk_assign(cand, Lm)]
    for k in range(1, members):
        res.append(perturbed(cand, Lm, 0.3, 1000 * k + s))
    return s, res


ref = np.load(os.path.join(K7, "links.npz"))["qn_ref"]
anch = np.zeros(N, bool)
for s in subs:
    anch[tabs[s]["loc"]] = tabs[s]["anch"]


def split(pred):
    off = 0; prs = {}
    for s in subs:
        k = len(tabs[s]["y"]); prs[s] = pred[off:off + k]; off += k
    return prs


def match_all(pred, members, label):
    prs = split(pred)
    out = Parallel(n_jobs=a.jobs)(delayed(match)(s, prs[s], members, ref) for s in subs)
    Su = np.full((members, N), -1, np.int64); Sc = np.full((members, N), -50.0, np.float32)
    for s, res in out:
        loc = tabs[s]["loc"]
        for k, (su, sc) in enumerate(res):
            Su[k, loc] = np.where(su >= 0, loc[np.maximum(su, 0)], -1); Sc[k, loc] = sc
    h = ts >= 0; e = Su[0][h] == ts[h]
    same = (sens == sens[np.maximum(ts, 0)])[h]; y_ = S["oof_y"][h]
    print(f"RESULT {label} chain={a.chain} anchor={a.anchor} W={a.W} err={a.chain_err} fits={a.fits}: exact successor {e.mean():.4f} | same-limb {e[same].mean():.4f} "
          f"cross-limb {e[~same].mean():.4f} | anchored A {e[anch[h]].mean():.4f} unanchored A {e[~anch[h]].mean() if (~anch[h]).any() else float('nan'):.4f} | "
          f"null tiles {e[y_ == 0].mean():.4f} activity tiles {e[y_ > 0].mean():.4f} [{time.time() - T0:.0f}s]", flush=True)
    print("   by fold:", " ".join(f"{np.mean(e[fold[h] == f_]):.4f}" for f_ in range(5)), flush=True)
    return Su, Sc


def gap_feats(s, su_glob):
    """stage 2: predicted paths (stage-1 plain matching) behind A and ahead of B; for every limb L the last own tile of L
    behind A (u steps) and the first own tile of L ahead of B (v steps) must be exactly 1+u+v steps apart on chain L, with
    no own tile of L in between"""
    t = tabs[s]; n = t["n"]; loc = t["loc"]; l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    su = su_glob[loc]; su = np.where(su >= 0, l_of[np.maximum(su, 0)], -1); pr = np.full(n, -1); pr[su[su >= 0]] = np.flatnonzero(su >= 0)
    Hh = 12; lim, anch_, pos = t["lim"], t["anch"], t["pos"]
    U = np.full((n, 4), 99); XU = np.full((n, 4), -1); V = np.full((n, 4), 99); YV = np.full((n, 4), -1)
    curb = np.arange(n); curf = np.arange(n)
    for k in range(Hh):
        for cur, Arr, Node in ((curb, U, XU), (curf, V, YV)):
            ok = cur >= 0; c = np.maximum(cur, 0); L = lim[c]; m = ok & anch_[c]
            sel = m & (Arr[np.arange(n), L] == 99); Arr[np.flatnonzero(sel), L[sel]] = k; Node[np.flatnonzero(sel), L[sel]] = c[sel]
        curb = np.where(curb >= 0, pr[np.maximum(curb, 0)], -1); curf = np.where(curf >= 0, su[np.maximum(curf, 0)], -1)
    pi, pj = t["pi"], t["pj"]; cons = np.zeros(len(pi), np.float32); inc = np.zeros(len(pi), np.float32); unk = np.zeros(len(pi), np.float32)
    ccons = np.zeros(len(pi), np.float32); cinc = np.zeros(len(pi), np.float32); own = {}
    for L in range(4):
        u, v, xn, yn = U[pi, L], V[pj, L], XU[pi, L], YV[pj, L]; known = (u < 99) & (v < 99)
        g = 1 + u + v; df = t["dfirst"][L][pos[np.maximum(xn, 0)]]; nf = t["nfirst"][L][pos[np.maximum(xn, 0)]]; cf = t["cfirst"][L][pos[np.maximum(xn, 0)]]
        ok = known & (df == g) & (nf == pos[np.maximum(yn, 0)]); bad = known & ~ok & ((df < 99) | (g <= a.W))
        cons += ok; inc += bad; unk += ~(ok | bad); ccons += ok * cf; cinc += bad * cf
        own[L] = np.where(ok, 1.0, np.where(bad, -1.0, 0.0))
    LA, LB = lim[pi], lim[pj]
    f_LA = np.choose(LA, [own[0], own[1], own[2], own[3]]); f_LB = np.choose(LB, [own[0], own[1], own[2], own[3]])
    return np.stack([cons, inc, unk, ccons, cinc, f_LA, f_LB], 1).astype(np.float32)


two = a.stage2 and a.chain != "none"
Su, Sc = match_all(pred, a.members if (a.save and not two) else 1, "stage1")
if two:
    names2 = names + ["gap_cons", "gap_inc", "gap_unk", "gap_ccons", "gap_cinc", "gap_LA", "gap_LB", "prev_logit"]
    for it in range(a.iters):
        G = Parallel(n_jobs=a.jobs)(delayed(gap_feats)(s, Su[0]) for s in subs)
        Xk = np.concatenate([X, np.concatenate(G), pred[:, None]], 1)
        pred = nested(Xk, names2, f"stage{it + 2}", base=pred); del Xk
        Su, Sc = match_all(pred, a.members if (a.save and it == a.iters - 1) else 1, f"stage{it + 2}")
if a.save:
    h = ts >= 0
    for k in range(a.members):
        Sc[k] = qnorm(Su[k], Sc[k], sbj, ref)
    print("   perturbed mean exact", np.mean([(Su[k][h] == ts[h]).mean() for k in range(1, a.members)]).round(4))
    tag = a.tag or f"{a.chain}_a{a.anchor}"
    np.savez(os.path.join(OUT, "cache", f"links_{tag}.npz"), oof_succ=Su, oof_score=Sc)
    print("saved", tag)
