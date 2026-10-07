"""Shared helpers for the test-side w25 chain pipeline (task D). Ports of the research2/w25feas simulation code
(chain_feats / train_chain / build_links), generalised so that every limb's chain has its own node set:
  pos[i]      index of our tile i in the chain of ITS OWN limb (valid only where anch[i])
  owner[L, q] link-local index of the anchored tile of limb L sitting at node q of chain L, else -1
  mark[L, q]  = owner[L, q] >= 0
In the training simulation all chains share the node set (positions = seconds); in the real 2025 data each limb's rows are
shuffled independently, so the same code is fed per-limb node indices. Nothing outside this folder is modified."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

W = r"E:\Claude code\wear"
FEAS = os.path.join(W, "exp", "v4", "research2", "w25feas")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); MODELS = os.path.join(HERE, "models")
os.makedirs(CACHE, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
sys.path.insert(0, FEAS); sys.path.insert(0, os.path.join(W, "exp", "v4"))
from common import load_stage, subject_tiles, SENS, K7, K9          # noqa: E402
from sim_chain import cost_matrix                                    # noqa: E402
_argv = sys.argv; sys.argv = [_argv[0]]                              # chain_feats parses argv at import
import chain_feats as _cf                                            # noqa: E402
sys.argv = _argv
tile_stats, pair_feats, FEATS = _cf.tile_stats, _cf.pair_feats, _cf.FEATS
from relink import lk_assign, perturbed, qnorm                       # noqa: E402

L25 = ["left_arm", "left_leg", "right_arm", "right_leg"]              # w25.npz limb code order (exp/hyb/w25_parse.py)
MAP25 = np.array([SENS.index(x) for x in L25])                       # 2025 limb code -> pipeline sensor index
FITS = {"K7": K7, "K9": K9}
# chain scorer (train_chain.py)
P_CHAIN = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, max_bin=63)
ROUNDS_CHAIN = 200
# link re-scorer (build_links.py)
P_LINK = dict(objective="binary", learning_rate=0.08, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3)
ROUNDS_LINK = 300
DROP = {"rk_row", "d_rowmax", "d_colmax"}
LLR = {(1, 0): np.log(4 / 3), (0, 1): np.log(4 / 3), (0, 0): np.log(8 / 9)}
GAP_NAMES = ["gap_cons", "gap_inc", "gap_unk", "gap_ccons", "gap_cinc", "gap_LA", "gap_LB", "prev_logit"]


# ----------------------------------------------------------------------------------------------- chains (per limb)
def chain_cand_feats(T, L, K=25, Kc=10):
    """chain_feats.run_subject inner loop on one limb's rows T (n,50,3): candidate pairs + 26 features"""
    n = len(T)
    C = cost_matrix(T, "nll3"); F = tile_stats(T)
    top = np.argpartition(C, K, 1)[:, :K]
    topc = np.argpartition(C, Kc, 0)[:Kc, :]
    pi = np.r_[np.repeat(np.arange(n), K), topc.ravel()]; pj = np.r_[top.ravel(), np.tile(np.arange(n), Kc)]
    key = np.unique(pi.astype(np.int64) * n + pj); pi, pj = key // n, key % n
    ok = pi != pj; pi, pj = pi[ok], pj[ok]
    cv = C[pi, pj]; rr = np.zeros(len(pi)); rc = np.zeros(len(pi))
    CT = np.ascontiguousarray(C.T)
    for b in range(0, n, 64):
        m = np.flatnonzero((pi >= b) & (pi < b + 64)); rr[m] = (C[pi[m]] < cv[m][:, None]).sum(1)
        m = np.flatnonzero((pj >= b) & (pj < b + 64)); rc[m] = (CT[pj[m]] < cv[m][:, None]).sum(1)
    del CT
    Cp = np.partition(C, 1, 1); rowbest, row2 = Cp[:, 0], Cp[:, 1]; del Cp; colbest = C.min(0)
    X = pair_feats(F, pi, pj, C, rr, rc, rowbest, row2, colbest, L)
    return pi.astype(np.int64), pj.astype(np.int64), X


def chain_assign(n, pi, pj, lo):
    """train_chain.assign_subject for one limb: kernel Hungarian on chain log-odds -> succ, conf"""
    cnt = np.bincount(pi, minlength=n); K = int(cnt.max()); order = np.argsort(pi, kind="stable")
    start = np.r_[0, np.cumsum(cnt)][:-1]; slot = np.arange(len(pi)) - start[pi[order]]
    cand = np.full((n, K), -1, np.int64); Lm = np.full((n, K), -50.0, np.float32)
    cand[pi[order], slot] = pj[order]; Lm[pi[order], slot] = lo[order]
    succ, sc = lk_assign(cand, Lm)
    return succ, np.where(succ >= 0, 1 / (1 + np.exp(-sc)), 0.0).astype(np.float32)


def score_chain_limb(T, L, model_file):
    import lightgbm as lgb
    pi, pj, X = chain_cand_feats(T, L)
    m = lgb.Booster(model_file=model_file)
    lo = m.predict(X, raw_score=True, num_threads=1).astype(np.float32)
    su, cf = chain_assign(len(T), pi, pj, lo)
    return L, su, cf


def chain_tuple(su, cf):
    """(succ, pred, conf of link out, conf of link in) as build_links.chain_for"""
    n = len(su); su = su.astype(np.int64); cf = cf.astype(np.float32)
    pr = np.full(n, -1, np.int64); pr[su[su >= 0]] = np.flatnonzero(su >= 0)
    cb = np.zeros(n, np.float32); cb[su[su >= 0]] = cf[su >= 0]
    return su, pr, cf, cb


# ----------------------------------------------------------------------------------------------- 2025 rows of a subject
def prepare_subject(s, ourX, lim, twin_rows, anch, A25, s25, pl25):
    """our tiles (link-local order) + their 2025 twins -> per-limb chain row sets with anchored twins replaced by our clean
    tiles, pos (index of each anchored tile in its own limb's chain) and owner (4, n)."""
    n = len(lim)
    rows_L = [np.flatnonzero((s25 == s) & (pl25 == L)) for L in range(4)]
    for L in range(4):
        assert len(rows_L[L]) == n, (s, L, len(rows_L[L]), n)
    where = np.full(len(s25), -1, np.int64)
    for L in range(4):
        where[rows_L[L]] = np.arange(n)
    a_idx = np.flatnonzero(anch)
    tw = twin_rows[a_idx]
    assert (tw >= 0).all() and (s25[tw] == s).all(), "twin of an anchored tile outside the subject"
    assert (pl25[tw] == lim[a_idx]).all(), "twin limb != our sensor"
    assert len(np.unique(tw)) == len(tw), "two anchored tiles share a twin"
    pos = np.zeros(n, np.int64); pos[a_idx] = where[tw]
    assert (pos[a_idx] >= 0).all()
    owner = np.full((4, n), -1, np.int64); owner[lim[a_idx], pos[a_idx]] = a_idx
    Ts = []
    for L in range(4):
        T = A25[rows_L[L]].astype(np.float32).copy()
        k = a_idx[lim[a_idx] == L]; T[pos[k]] = ourX[k].astype(np.float32)
        Ts.append(T)
    return dict(rows_L=rows_L, pos=pos, owner=owner, Ts=Ts)


# ----------------------------------------------------------------------------------------------- link tables (build_links port)
def walks(su, cf, n, Wn):
    F = np.full((n, Wn), -1, np.int64); C = np.zeros((n, Wn), np.float32); cur = np.arange(n); c = np.ones(n, np.float32)
    for k in range(Wn):
        ok = cur >= 0; nx = np.full(n, -1, np.int64); nx[ok] = su[cur[ok]]; c = np.where(nx >= 0, c * np.where(ok, cf[np.maximum(cur, 0)], 0), 0)
        F[:, k] = nx; C[:, k] = c; cur = nx
    return F, C


def topk_cands(cand, Lo, topk=50):
    cand = cand.astype(np.int64); Lo = Lo.astype(np.float32)
    if topk and cand.shape[1] > topk:
        o = np.argsort(-np.where((cand >= 0) & (Lo > -49), Lo, -1e9), 1)[:, :topk]
        cand, Lo = np.take_along_axis(cand, o, 1), np.take_along_axis(Lo, o, 1)
    return cand, Lo


def subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, tl, Wn=16):
    """build_links.subject_table with per-limb node sets. chains: list of 4 (su, pr, cf, cb) over each chain's nodes
    (all chains have n nodes); cand/Lo: L3 candidates (link-local, already top-k); tl: true successor (link-local) or -1."""
    t0 = time.time()
    mark = (owner >= 0).astype(np.int8)
    ch = chains
    if ch is not None:
        FW = [walks(c[0], c[2], n, Wn) for c in ch]; BW = [walks(c[1], c[3], n, Wn) for c in ch]
        extra = []
        for L in range(4):
            for i in np.flatnonzero((lim == L) & anch):
                q = ch[L][0][pos[i]]
                if q >= 0 and mark[L, q]:
                    extra.append((i, owner[L, q]))
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
    extra_out = {}
    if ch is not None:
        LA, LB = lim[pi], lim[pj]; pA, pB = pos[pi], pos[pj]; aA, aB = anch[pi], anch[pj]
        na = len(pi)
        mA_f = np.zeros((na, Wn), np.int8); cA_f = np.zeros((na, Wn), np.float32); mB_f = np.zeros((na, Wn), np.int8); cB_f = np.zeros((na, Wn), np.float32)
        mA_b = np.zeros((na, Wn), np.int8); cA_b = np.zeros((na, Wn), np.float32); mB_b = np.zeros((na, Wn), np.int8); cB_b = np.zeros((na, Wn), np.float32)
        nodeA1 = np.full(na, -1); cf1 = np.zeros(na, np.float32)
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
        same = LA == LB
        dirsame = (same & (nodeA1 == pB)).astype(np.float32)                 # same limb -> same chain: node ids comparable
        okA = aA.astype(np.float32)[:, None]; okB = aB.astype(np.float32)[:, None]
        mAll = np.concatenate([mA_b[:, ::-1], mA_f], 1).astype(np.float32); mBll = np.concatenate([mB_b[:, ::-1], mB_f], 1).astype(np.float32)
        cAll = np.concatenate([cA_b[:, ::-1], cA_f], 1) * okA; cBll = np.concatenate([cB_b[:, ::-1], cB_f], 1) * okB
        w = cAll * cBll * (~same)[:, None]
        conf = mAll * mBll
        llr = np.where(conf > 0, -3.0, np.where(mAll + mBll > 0, LLR[(1, 0)], LLR[(0, 0)]))
        ctr = Wn
        for h in (1, 2, 4, 8, 16):
            if h > Wn:
                continue
            sl = slice(ctr - h, ctr + h)
            f[f"conf_w{h}"] = (conf[:, sl] * w[:, sl]).sum(1); f[f"llr_w{h}"] = (llr[:, sl] * w[:, sl]).sum(1); f[f"wsum_w{h}"] = w[:, sl].sum(1)
        f["dirsame"] = np.where(same & aA, dirsame, np.nan).astype(np.float32)
        f["cf1"] = np.where(aA, cf1, np.nan).astype(np.float32)
        nA1m = np.where(nodeA1 >= 0, mark[LA, np.maximum(nodeA1, 0)], 0)     # node of chain LA
        # cross-limb: a marked node of chain LA holds a tile of limb LA, never B -> "!= B" holds automatically
        f["succ_is_other_own"] = np.where(aA, (nA1m == 1) & ~(same & (nodeA1 == pB)), np.nan).astype(np.float32)
        dA = np.where(mA_f.any(1), mA_f.argmax(1) + 1, 99); dB = np.where(mB_b.any(1), mB_b.argmax(1) + 1, 99)
        C_ = np.full(na, -1, np.int64); D_ = np.full(na, -1, np.int64)
        for L in range(4):
            mA = np.flatnonzero((LA == L) & (dA <= Wn)); nc = FW[L][0][pA[mA], dA[mA] - 1]
            C_[mA] = np.where(nc >= 0, owner[L][np.maximum(nc, 0)], -1)
            mB = np.flatnonzero((LB == L) & (dB <= Wn)); nd = BW[L][0][pB[mB], dB[mB] - 1]
            D_[mB] = np.where(nd >= 0, owner[L][np.maximum(nd, 0)], -1)
        f["dA"] = np.where(aA & ~same, dA, np.nan).astype(np.float32); f["dB"] = np.where(aB & ~same, dB, np.nan).astype(np.float32)
        f["bridge_BC"] = np.where(aA & ~same & (dA == 2), l3(pj, C_), np.nan).astype(np.float32)
        f["bridge_DA"] = np.where(aB & ~same & (dB == 2), l3(D_, pi), np.nan).astype(np.float32)
        f["anch_A"] = aA.astype(np.float32); f["anch_B"] = aB.astype(np.float32)
        dfirst = np.full((4, n), 99, np.int64); nfirst = np.full((4, n), -1, np.int64); cfirst = np.zeros((4, n), np.float32)
        for L in range(4):
            Fn, Fc = FW[L]; mk = np.where(Fn >= 0, mark[L][np.maximum(Fn, 0)], 0); has = mk.any(1); k = mk.argmax(1)
            dfirst[L, has] = k[has] + 1; nfirst[L, has] = Fn[has, k[has]]; cfirst[L, has] = Fc[has, k[has]]
        extra_out = dict(dfirst=dfirst, nfirst=nfirst, cfirst=cfirst)
    names = list(f.keys()); X = np.stack([np.asarray(f[k], np.float32) for k in names], 1)
    yv = (tl[pi] == pj).astype(np.int8)
    return dict(s=s, pi=pi, pj=pj, X=X, y=yv, names=names, tl=tl, n=n, lim=lim, anch=anch, pos=pos, t=time.time() - t0, **extra_out)


def gap_feats(t, su, Wn=16):
    """build_links.gap_feats; su = link-local plain matching successor of the subject"""
    n = t["n"]; su = su.astype(np.int64)
    pr = np.full(n, -1); pr[su[su >= 0]] = np.flatnonzero(su >= 0)
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
        g = 1 + u + v; px = pos[np.maximum(xn, 0)]                           # xn, yn are anchored tiles of limb L -> chain-L nodes
        df = t["dfirst"][L][px]; nf = t["nfirst"][L][px]; cf = t["cfirst"][L][px]
        ok = known & (df == g) & (nf == pos[np.maximum(yn, 0)]); bad = known & ~ok & ((df < 99) | (g <= Wn))
        cons += ok; inc += bad; unk += ~(ok | bad); ccons += ok * cf; cinc += bad * cf
        own[L] = np.where(ok, 1.0, np.where(bad, -1.0, 0.0))
    LA, LB = lim[pi], lim[pj]
    f_LA = np.choose(LA, [own[0], own[1], own[2], own[3]]); f_LB = np.choose(LB, [own[0], own[1], own[2], own[3]])
    return np.stack([cons, inc, unk, ccons, cinc, f_LA, f_LB], 1).astype(np.float32)


def match(t, pr, members, s, tau=0.3):
    n = t["n"]; pi, pj = t["pi"], t["pj"]
    cnt = np.bincount(pi, minlength=n); K = int(cnt.max()); order = np.argsort(pi, kind="stable")
    start = np.r_[0, np.cumsum(cnt)][:-1]; slot = np.arange(len(pi)) - start[pi[order]]
    cand = np.full((n, K), -1, np.int64); Lm = np.full((n, K), -50.0, np.float32)
    cand[pi[order], slot] = pj[order]; Lm[pi[order], slot] = np.clip(pr[order], -45, 45)
    res = [lk_assign(cand, Lm)]
    for k in range(1, members):
        res.append(perturbed(cand, Lm, tau, 1000 * k + s))
    return s, res


def feat_cols(names):
    return [k for k, nm in enumerate(names) if nm not in DROP]
