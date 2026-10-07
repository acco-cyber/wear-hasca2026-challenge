"""Unit check: tdlib.subject_table + gap_feats (per-limb node sets) reproduce build_links.subject_table / gap_feats exactly
in the simulation setting (common node set), for anchor 1.0 and anchor 0.6. The reference functions below are verbatim
copies from research2/w25feas/build_links.py (that file runs at import, so it cannot be imported)."""
import os, sys, time, types
import numpy as np
from tdlib import *

S = load_stage(K7); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]; N = len(sbj)
OUT = FEAS
a = types.SimpleNamespace(chain="lgb", anchor=1.0, W=16, topk=50)
LZ_PATHS = [os.path.join(K7, "link_logodds.npz")]


# ------------------------------------------------------------------ verbatim reference (build_links.py)
def chain_for(s, rows, n):
    rec = S["oof_rec"][rows]; nxt = np.where(np.r_[rec[1:] == rec[:-1], False], np.arange(1, n + 1), -1)
    out = []
    z = np.load(os.path.join(OUT, "cache", f"chain_{a.chain}_s{s}.npz")); assert (z["rows"] == rows).all()
    for L in range(4):
        su = z[f"succ{L}"].astype(np.int64)
        cf = z[f"conf{L}"].astype(np.float32) if f"conf{L}" in z.files else np.ones(n, np.float32)
        out.append((su, cf))
    res = []
    for su, cf in out:
        pr = np.full(n, -1, np.int64); pr[su[su >= 0]] = np.flatnonzero(su >= 0)
        cb = np.zeros(n, np.float32); cb[su[su >= 0]] = cf[su >= 0]
        res.append((su, pr, cf, cb))
    return res


def ref_subject_table(s):
    loc = np.flatnonzero(sbj == s); n = len(loc)
    rows = loc[np.lexsort((S["oof_start"][loc], S["oof_rec"][loc]))]
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    pos = p_of[loc]; owner = l_of[rows]; lim = sens[loc]
    rng = np.random.default_rng(s); anch = rng.random(n) < a.anchor
    mark = np.zeros((4, n), np.int8)
    for L in range(4):
        mark[L, pos[(lim == L) & anch]] = 1
    tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
    LZ = [np.load(p) for p in LZ_PATHS]
    cand, Lo = LZ[0][f"oof_{s}_cand"].astype(np.int64), LZ[0][f"oof_{s}_L"].astype(np.float32)
    if a.topk and cand.shape[1] > a.topk:
        o = np.argsort(-np.where((cand >= 0) & (Lo > -49), Lo, -1e9), 1)[:, :a.topk]
        cand, Lo = np.take_along_axis(cand, o, 1), np.take_along_axis(Lo, o, 1)
    ch = chain_for(s, rows, n)
    Wn = a.W
    FW = [walks(c[0], c[2], n, Wn) for c in ch]; BW = [walks(c[1], c[3], n, Wn) for c in ch]
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
    rk_row = (np.argsort(np.argsort(-np.where(valid, Lo, -np.inf), 1), 1))[pi, ci]
    key = pi * n + pj; order = np.argsort(key); skey = key[order]; slo = lo[order]

    def l3(x, y_):
        k = x * n + y_; ix = np.clip(np.searchsorted(skey, k), 0, len(skey) - 1); hit = (skey[ix] == k) & (x >= 0) & (y_ >= 0)
        return np.where(hit, slo[ix], np.nan)
    colmax = np.full(n, -np.inf); np.maximum.at(colmax, pj, lo)
    rowmax = np.where(valid, Lo, -np.inf).max(1)
    f = {"lo": lo, "rk_row": rk_row.astype(np.float32), "d_rowmax": lo - rowmax[pi], "d_colmax": lo - colmax[pj],
         "same_limb": (lim[pi] == lim[pj]).astype(np.float32)}
    LA, LB = lim[pi], lim[pj]; pA, pB = pos[pi], pos[pj]; aA, aB = anch[pi], anch[pj]
    na = len(pi)
    mA_f = np.zeros((na, Wn), np.int8); cA_f = np.zeros((na, Wn), np.float32); mB_f = np.zeros((na, Wn), np.int8); cB_f = np.zeros((na, Wn), np.float32)
    mA_b = np.zeros((na, Wn), np.int8); cA_b = np.zeros((na, Wn), np.float32); mB_b = np.zeros((na, Wn), np.int8); cB_b = np.zeros((na, Wn), np.float32)
    nodeA1 = np.full(na, -1); nodeB0 = np.full(na, -1); dirsame = np.zeros(na, np.float32); cf1 = np.zeros(na, np.float32)
    for L in range(4):
        mA = LA == L
        Fn, Fc = FW[L][0][pA[mA]], FW[L][1][pA[mA]]; Bn, Bc = BW[L][0][pA[mA]], BW[L][1][pA[mA]]
        mA_f[mA] = np.where(Fn >= 0, mark[L][np.maximum(Fn, 0)], 0); cA_f[mA] = Fc
        mA_b[mA, 0] = 1; cA_b[mA, 0] = 1.0
        mA_b[mA, 1:] = np.where(Bn[:, :-1] >= 0, mark[L][np.maximum(Bn[:, :-1], 0)], 0); cA_b[mA, 1:] = Bc[:, :-1]
        nodeA1[mA] = Fn[:, 0]; cf1[mA] = Fc[:, 0]
        mB = LB == L
        Fn, Fc = FW[L][0][pB[mB]], FW[L][1][pB[mB]]; Bn, Bc = BW[L][0][pB[mB]], BW[L][1][pB[mB]]
        mB_f[mB, 0] = 1; cB_f[mB, 0] = 1.0
        mB_f[mB, 1:] = np.where(Fn[:, :-1] >= 0, mark[L][np.maximum(Fn[:, :-1], 0)], 0); cB_f[mB, 1:] = Fc[:, :-1]
        mB_b[mB] = np.where(Bn >= 0, mark[L][np.maximum(Bn, 0)], 0); cB_b[mB] = Bc
        nodeB0[mB] = Bn[:, 0]
    same = LA == LB
    dirsame = (same & (nodeA1 == pB)).astype(np.float32)
    okA = aA.astype(np.float32)[:, None]; okB = aB.astype(np.float32)[:, None]
    mAll = np.concatenate([mA_b[:, ::-1], mA_f], 1).astype(np.float32); mBll = np.concatenate([mB_b[:, ::-1], mB_f], 1).astype(np.float32)
    cAll = np.concatenate([cA_b[:, ::-1], cA_f], 1) * okA; cBll = np.concatenate([cB_b[:, ::-1], cB_f], 1) * okB
    w = cAll * cBll * (~same)[:, None]
    conf = mAll * mBll
    llr = np.where(conf > 0, -3.0, np.where(mAll + mBll > 0, LLR[(1, 0)], LLR[(0, 0)]))
    ctr = Wn
    for h in (1, 2, 4, 8, 16):
        sl = slice(ctr - h, ctr + h)
        f[f"conf_w{h}"] = (conf[:, sl] * w[:, sl]).sum(1); f[f"llr_w{h}"] = (llr[:, sl] * w[:, sl]).sum(1); f[f"wsum_w{h}"] = w[:, sl].sum(1)
    f["dirsame"] = np.where(same & aA, dirsame, np.nan).astype(np.float32)
    f["cf1"] = np.where(aA, cf1, np.nan).astype(np.float32)
    nA1m = np.where(nodeA1 >= 0, mark[LA, np.maximum(nodeA1, 0)], 0)
    f["succ_is_other_own"] = np.where(aA, (nA1m == 1) & (nodeA1 != pB), np.nan).astype(np.float32)
    dA = np.where(mA_f.any(1), mA_f.argmax(1) + 1, 99); dB = np.where(mB_b.any(1), mB_b.argmax(1) + 1, 99)
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
    dfirst = np.full((4, n), 99, np.int64); nfirst = np.full((4, n), -1, np.int64); cfirst = np.zeros((4, n), np.float32)
    for L in range(4):
        Fn, Fc = FW[L]; mk = np.where(Fn >= 0, mark[L][np.maximum(Fn, 0)], 0); has = mk.any(1); k = mk.argmax(1)
        dfirst[L, has] = k[has] + 1; nfirst[L, has] = Fn[has, k[has]]; cfirst[L, has] = Fc[has, k[has]]
    return dict(s=s, loc=loc, pi=pi, pj=pj, X=X, names=names, tl=tl, n=n, lim=lim, anch=anch, dfirst=dfirst, nfirst=nfirst, cfirst=cfirst, pos=pos, owner=owner)


def ref_gap(t, su):
    n = t["n"]; pr = np.full(n, -1); pr[su[su >= 0]] = np.flatnonzero(su >= 0)
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


# ------------------------------------------------------------------ port under test (the same inputs as train_tables.sim_inputs)
def port(s):
    loc = np.flatnonzero(sbj == s); n = len(loc)
    rows = loc[np.lexsort((S["oof_start"][loc], S["oof_rec"][loc]))]
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    pos = p_of[loc]; lim = sens[loc]
    anch = np.random.default_rng(s).random(n) < a.anchor
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
    z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); assert (z["rows"] == rows).all()
    chains = [chain_tuple(z[f"succ{L}"], z[f"conf{L}"]) for L in range(4)]
    LZ = np.load(os.path.join(K7, "link_logodds.npz"))
    cand, Lo = topk_cands(LZ[f"oof_{s}_cand"], LZ[f"oof_{s}_L"], 50)
    return subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, tl)


if __name__ == "__main__":
    for anc in (1.0, 0.6):
        a.anchor = anc
        for s in (5, 20):
            r = ref_subject_table(s); p = port(s)
            assert r["names"] == p["names"], (r["names"], p["names"])
            assert (r["pi"] == p["pi"]).all() and (r["pj"] == p["pj"]).all()
            same = np.array_equal(r["X"], p["X"], equal_nan=True)
            for key in ("dfirst", "nfirst", "cfirst"):
                assert np.array_equal(r[key], p[key]), key
            rng = np.random.default_rng(s); su = np.full(p["n"], -1)                      # a random partial matching for the gap features
            perm = rng.permutation(p["n"]); su[perm[:-1]] = perm[1:]
            g_same = np.array_equal(ref_gap(r, su), gap_feats(p, su), equal_nan=True)
            print(f"anchor {anc} sbj {s}: pairs {len(p['pi'])} X identical {same} gap identical {g_same}", flush=True)
            assert same and g_same
    print("PORT CHECK OK")
