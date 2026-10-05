"""Boundary refiner with a bout-transition prior (key: transition).

Fused K7+K9 decode (cache.npz from fuse.py) -> the kernel's boundary refiner (rows re-implemented from
v4_local.refiner_rows, identical base features) plus features from an empirical bout-transition model
P(next activity B | previous activity A) over non-null bouts (null gaps ignored, repeats merged), counted from the
TRUE label order (oof_rec, oof_start) of training subjects only.

Nesting: the model evaluated on fold k is trained on rows of folds j != k whose transition features come from
T_{-{j,k}} (counts without fold j AND fold k); fold-k rows use T_{-k}. Test: rows of fold j use T_{-j}, test rows T_all.

Feature sets (--feat):
  base   : kernel refiner features only (reproduces the 0.9330 baseline)
  pair   : base + boundary-pair transition features: log P(B|A), log P(A|B) (backward model), each also relative to
           the best alternative, for the label change A(i) -> B(succ i), nulls resolved to the adjacent non-null bout
  ctx    : pair + bout context: for the flipped tile's bout (cur) and the bout across the boundary (oth):
           log P(cur|prev)+log P(next|cur), log P(next|prev) (bout removed), their difference; bout / run lengths
  len    : base + bout / run lengths only (control without the transition prior)
  direct : base + pair features only for act|act changes (raw labels, no null resolution through the chain)
  python refine_tr.py --feat base,pair,ctx,len,direct [--alpha 0.5] [--jobs 3] [--seed S] [--write KIND]
Result (2026-10-05): no variant beats the base refiner (0.9330); see run*.log."""
import os, sys, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
from v4_local import macro_f1, N_CLS, FOLDS, log, write_sub, refiner_flips, REF_H, REF_THR, REF_ROUNDS, REF_PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
MAXWALK = 10 ** 9


# ------------------------------------------------------------------ chain context (decoded labels along a matching)
def chain_context(fin, succ):
    """prevB / nextB: label of the previous / next non-null bout along the matching chain (null gaps ignored, repeats
    merged); for a null tile, the last non-null label before / first after its gap. runlen: run of identical labels
    containing the tile; boutlen: non-null tiles of the tile's (null-merged) bout, or the null gap length."""
    n = len(fin)
    prv = np.full(n, -1); prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
    seen = np.zeros(n, bool)
    prevB = np.full(n, -1); nextB = np.full(n, -1); runlen = np.zeros(n, np.int32); boutlen = np.zeros(n, np.int32)
    heads = list(np.flatnonzero(prv < 0))
    paths = []
    for h in heads:
        p = []; g = h
        while g >= 0 and not seen[g]:
            seen[g] = True; p.append(g); g = succ[g]
        paths.append(p)
    for g0 in np.flatnonzero(~seen):              # cycles: break at an arbitrary tile
        if seen[g0]:
            continue
        p = []; g = g0
        while g >= 0 and not seen[g]:
            seen[g] = True; p.append(g); g = succ[g]
        paths.append(p)
    for p in paths:
        p = np.array(p); L = fin[p]; m = len(p)
        # forward: previous non-null bout
        lastNN, boutPrev = -1, -1; bid = np.zeros(m, np.int64); b = -1
        pb = np.full(m, -1)
        for t in range(m):
            c = L[t]
            if c == 0:
                pb[t] = lastNN
            else:
                if c != lastNN:
                    boutPrev = lastNN; lastNN = c; b += 1
                pb[t] = boutPrev
            bid[t] = b
        # backward: next non-null bout
        lastNN, boutNext = -1, -1; nb = np.full(m, -1)
        for t in range(m - 1, -1, -1):
            c = L[t]
            if c == 0:
                nb[t] = lastNN
            else:
                if c != lastNN:
                    boutNext = lastNN; lastNN = c
                nb[t] = boutNext
        # run lengths of identical labels
        ch = np.r_[0, np.flatnonzero(np.diff(L) != 0) + 1, m]
        rl = np.repeat(np.diff(ch), np.diff(ch))
        # bout lengths: non-null tiles per null-merged bout; null tiles get their gap (run) length
        nn = L != 0
        bl = rl.copy()
        if nn.any():
            cnt = np.bincount(bid[nn] + 1); bl[nn] = cnt[bid[nn] + 1]
        prevB[p], nextB[p], runlen[p], boutlen[p] = pb, nb, rl, bl
    return prevB, nextB, runlen, boutlen


def refiner_rows_ctx(fin, succ, B, P, Q, lw, sens, ener, post, vmot, vmean):
    """v4_local.refiner_rows (identical base features, identical row order) + integer context per row"""
    n = len(fin)
    prv = np.full(n, -1); prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
    lP, lQ = np.log(np.clip(P, 1e-9, None)), np.log(Q + 1e-9)
    prevB, nextB, runlen, boutlen = chain_context(fin, succ)

    def step(g, k):
        for _ in range(abs(k)):
            g = succ[g] if k > 0 else prv[g]
            if g < 0:
                return -1
        return g

    rows, tiles, others, ctx = [], [], [], []
    for i in np.flatnonzero((succ >= 0) & (fin != fin[np.maximum(succ, 0)])):
        j = succ[i]; A, Bc = fin[i], fin[j]
        An = A if A != 0 else prevB[i]; Bn = Bc if Bc != 0 else nextB[j]     # boundary pair, nulls resolved
        for off in range(-REF_H + 1, REF_H + 1):
            g = step(i, off) if off <= 0 else step(j, off - 1)
            if g < 0:
                continue
            cur, oth = fin[g], (Bc if off <= 0 else A)
            if cur == oth:
                continue
            o = j if off <= 0 else i                                       # tile across the boundary (label oth)
            nxt, prv_ = succ[g], prv[g]
            f = [off, int(cur == 0), int(oth == 0), B[g, cur] - B[g, oth], lP[g, cur] - lP[g, oth], lw[g, cur] - lw[g, oth], lQ[g, cur] - lQ[g, oth],
                 B[g, 0], lP[g, 0], ener[g], vmot[g],
                 float(np.linalg.norm(vmean[g] - vmean[prv_])) if prv_ >= 0 else -1.0, float(np.linalg.norm(vmean[g] - vmean[nxt])) if nxt >= 0 else -1.0]
            for nb in (-2, -1, 1, 2):
                gj = step(g, nb)
                if gj >= 0:
                    f += [B[gj, cur] - B[gj, oth], lP[gj, cur] - lP[gj, oth], ener[gj] - ener[g], int(sens[gj] == sens[g]),
                          float(np.linalg.norm(post[gj] - post[g])) if sens[gj] == sens[g] else -1.0, int(fin[gj] == cur)]
                else:
                    f += [0, 0, 0, 0, -1.0, -1]
            rows.append(f); tiles.append(g); others.append(oth)
            ctx.append([An, Bn, cur, oth, prevB[g], nextB[g], prevB[o], nextB[o], runlen[g], boutlen[g], runlen[o], boutlen[o], A, Bc])
    return np.array(rows, np.float32), np.array(tiles, np.int64), np.array(others, np.int64), np.array(ctx, np.int64)


# ------------------------------------------------------------------ transition model
def bout_sequences(y, rec, start, keep):
    seqs = []
    for r in np.unique(rec[keep]):
        ii = np.flatnonzero((rec == r) & keep); o = ii[np.argsort(start[ii])]; lab = y[o]; lab = lab[lab != 0]
        if len(lab):
            seqs.append(lab[np.r_[True, np.diff(lab) != 0]])
    return seqs


def trans_logp(seqs, alpha):
    C = np.zeros((N_CLS, N_CLS))
    for s in seqs:
        np.add.at(C, (s[:-1], s[1:]), 1)
    C[0, :] = 0; C[:, 0] = 0; np.fill_diagonal(C, 0)
    act = np.arange(1, N_CLS); M = np.zeros((N_CLS, N_CLS)); M[np.ix_(act, act)] = 1; np.fill_diagonal(M, 0)
    Cs = (C + alpha) * M
    lf = np.log(Cs / Cs.sum(1, keepdims=True).clip(1e-12) + 1e-12)       # P(next=b | prev=a)
    lb = np.log(Cs / Cs.sum(0, keepdims=True).clip(1e-12) + 1e-12)       # P(prev=a | next=b)
    lf[~M.astype(bool)] = 0; lb[~M.astype(bool)] = 0
    return lf, lb


def trans_feats(ctx, lf, lb, kind):
    """features from the integer context and one transition model (lf/lb indexed [prev, next])"""
    An, Bn, cur, oth, pbg, nbg, pbo, nbo, rlg, blg, rlo, blo = ctx.T[:12]
    nanv = np.nan
    if kind == "direct":                                  # raw boundary pair, only act|act changes (no chain walk)
        Ar, Br = ctx[:, 12], ctx[:, 13]; ok = (Ar > 0) & (Br > 0)
        lfm = np.where(lf == 0, -np.inf, lf); lbm = np.where(lb == 0, -np.inf, lb)
        a_, b_ = np.clip(Ar, 0, None), np.clip(Br, 0, None)
        fAB = np.where(ok, lf[a_, b_], nanv); bAB = np.where(ok, lb[a_, b_], nanv)
        return np.stack([fAB, bAB, np.where(ok, fAB - lfm.max(1)[a_], nanv), np.where(ok, bAB - lbm.max(0)[b_], nanv)], 1).astype(np.float32)
    lenf = [rlg, np.log1p(blg), rlo, np.log1p(blo)]
    if kind == "len":
        return np.stack(lenf, 1).astype(np.float32)

    def look(T, a, b):                                    # NaN where a context label is missing; 0 for a == b (merge)
        v = T[np.clip(a, 0, None), np.clip(b, 0, None)].astype(np.float64)
        v[(a < 0) | (b < 0)] = nanv
        return v
    okp = (An > 0) & (Bn > 0) & (An != Bn)
    fAB = np.where(okp, look(lf, An, Bn), nanv); bAB = np.where(okp, look(lb, An, Bn), nanv)
    lfm = np.where(lf == 0, -np.inf, lf); lbm = np.where(lb == 0, -np.inf, lb)
    rfAB = np.where(okp, fAB - lfm.max(1)[np.clip(An, 0, None)], nanv)              # vs best next for A
    rbAB = np.where(okp, bAB - lbm.max(0)[np.clip(Bn, 0, None)], nanv)              # vs best prev for B
    same = ((An == Bn) & (An > 0)).astype(np.float64)
    pair = [fAB, bAB, rfAB, rbAB, same]
    if kind == "pair":
        return np.stack(pair, 1).astype(np.float32)

    def bout_fit(lab, pb, nb):
        """log P(lab|prev) + log P(next|lab), log P(next|prev) (bout removed), gain of removing it"""
        ok = lab > 0
        a1 = np.where(ok & (pb > 0) & (pb != lab), look(lf, pb, lab), nanv)
        a2 = np.where(ok & (nb > 0) & (nb != lab), look(lf, lab, nb), nanv)
        fin_ = np.where(np.isnan(a1), 0, a1) + np.where(np.isnan(a2), 0, a2)
        fin_[np.isnan(a1) & np.isnan(a2)] = nanv
        sk = np.where((pb > 0) & (nb > 0), np.where(pb == nb, 0.0, look(lf, pb, nb)), nanv)
        return [fin_, sk, sk - fin_]
    ctxf = bout_fit(cur, pbg, nbg) + bout_fit(oth, pbo, nbo)
    return np.stack(pair + ctxf + lenf, 1).astype(np.float32)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat", default="base,pair,ctx,len"); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--write", default="")
    ap.add_argument("--thr", type=float, default=REF_THR)
    ap.add_argument("--seed", type=int, default=-1, help="LightGBM master seed (-1: library default, as the kernel)")
    a = ap.parse_args()
    import lightgbm as lgb
    prm = dict(REF_PARAMS, num_threads=a.threads)
    sfx = ""
    if a.seed >= 0:
        prm["seed"] = a.seed; sfx = f"_s{a.seed}"
    z = np.load(os.path.join(HERE, "cache.npz"), allow_pickle=True)
    y, fold, rec, start = z["y"], z["fold"], z["oof_rec"], z["oof_start"]
    fin_o, fin_t = z["fin_o"], z["fin_t"]
    sc = np.load(os.path.join(K7, "tile_scalars.npz"))
    sc_o = (sc["oof_ener"], sc["oof_post"], sc["oof_vmot"], sc["oof_vmean"].astype(np.float32))
    sc_t = (sc["test_ener"], sc["test_post"], sc["test_vmot"], sc["test_vmean"].astype(np.float32))
    so, st_ = z["sensor_oof"].astype(np.int64), z["sensor_test"].astype(np.int64)
    Lo = list(z["oof_succ"]); Lt = list(z["test_succ"])
    log(f"pre-refiner OOF F1 {macro_f1(y, fin_o):.4f}; {len(Lo)} matchings")
    cp = os.path.join(HERE, "rows_cache2.npz")
    if os.path.exists(cp):
        rc = np.load(cp); X, tl, ot, CX = rc["X"], rc["tl"], rc["ot"], rc["CX"]; Xt, tlt, ott, CXt = rc["Xt"], rc["tlt"], rc["ott"], rc["CXt"]
        log("rows from cache")
    else:
        ro = Parallel(n_jobs=a.jobs)(delayed(refiner_rows_ctx)(fin_o, su, z["Bo"], z["Po"], z["Qo"], z["lwo"], so, *sc_o) for su in Lo)
        X = np.concatenate([r[0] for r in ro]); tl = np.concatenate([r[1] for r in ro]); ot = np.concatenate([r[2] for r in ro]); CX = np.concatenate([r[3] for r in ro]); del ro
        rt = Parallel(n_jobs=a.jobs)(delayed(refiner_rows_ctx)(fin_t, su, z["Bt"], z["Pt"], z["Qt"], z["lwt"], st_, *sc_t) for su in Lt)
        Xt = np.concatenate([r[0] for r in rt]); tlt = np.concatenate([r[1] for r in rt]); ott = np.concatenate([r[2] for r in rt]); CXt = np.concatenate([r[3] for r in rt]); del rt
        np.savez(cp, X=X, tl=tl, ot=ot, CX=CX, Xt=Xt, tlt=tlt, ott=ott, CXt=CXt)
        log(f"rows built: {len(X)} oof, {len(Xt)} test")
    T = ((y[tl] == ot) & (y[tl] != fin_o[tl])).astype(int); Fr = fold[tl]
    log(f"{len(T)} rows, {T.mean():.4f} should flip")
    # transition models: T_{-k}, T_{-{j,k}}, T_all
    allk = np.arange(FOLDS)
    tm = {}
    for ex in [()] + [(k,) for k in allk] + [(j, k) for j in allk for k in allk if j < k]:
        keep = ~np.isin(fold, ex)
        tm[ex] = trans_logp(bout_sequences(y, rec, start, keep), a.alpha)
    TM = lambda *ex: tm[tuple(sorted(set(ex)))]
    lf_all = tm[()][0]
    names = ["jog", "jog-arms", "jog-skip", "jog-side", "jog-butt", "str-tri", "str-lunge", "str-shoul", "str-ham", "str-lumbar",
             "push", "push-c", "sit", "sit-c", "burpee", "lunge", "lunge-c", "dips"]
    for a_ in range(1, N_CLS):
        top = np.argsort(-lf_all[a_])[:3]
        log(f"  T_all: {a_:2d} -> " + ", ".join(f"{b} ({np.exp(lf_all[a_, b]):.2f})" for b in top if lf_all[a_, b] != 0))
    results = {}
    pb_ = os.path.join(HERE, f"pr_base{sfx}.npy")
    if "base" not in a.feat.split(",") and os.path.exists(pb_):          # baseline predictions from an earlier run
        pr = np.load(pb_).astype(np.float64); assert len(pr) == len(T)
        ref_o, _ = refiner_flips(fin_o, tl, ot, pr, len(Lo), a.thr)
        results["base"] = (macro_f1(y, ref_o), [macro_f1(y[fold == f], ref_o[fold == f]) for f in range(FOLDS)], ref_o, pr)
        log(f"[base] (cached predictions) F1 {results['base'][0]:.4f} | per fold " + " ".join(f"{v:.4f}" for v in results["base"][1]))
    for kind in a.feat.split(","):
        pr = np.zeros(len(T))
        for k in range(FOLDS):
            tr = Fr != k
            if kind == "base":
                Xtr, Xev = X[tr], X[~tr]
            else:
                Ftr = np.zeros((tr.sum(), trans_feats(CX[:2], *TM(0), kind).shape[1]), np.float32)
                idx_tr = np.flatnonzero(tr)
                for j in allk:
                    if j == k:
                        continue
                    m = Fr[idx_tr] == j
                    Ftr[m] = trans_feats(CX[idx_tr[m]], *TM(j, k), kind)
                Xtr = np.concatenate([X[tr], Ftr], 1)
                Xev = np.concatenate([X[~tr], trans_feats(CX[~tr], *TM(k), kind)], 1)
            pr[~tr] = lgb.train(prm, lgb.Dataset(Xtr, T[tr]), REF_ROUNDS).predict(Xev)
        ref_o, nf = refiner_flips(fin_o, tl, ot, pr, len(Lo), a.thr)
        pf = [macro_f1(y[fold == f], ref_o[fold == f]) for f in range(FOLDS)]
        results[kind] = (macro_f1(y, ref_o), pf, ref_o, pr)
        log(f"[{kind}] {nf} OOF tiles flipped, F1 {macro_f1(y, fin_o):.4f} -> {results[kind][0]:.4f} | per fold " + " ".join(f"{v:.4f}" for v in pf))
        np.save(os.path.join(HERE, f"pr_{kind}{sfx}.npy"), pr.astype(np.float32))
    if "base" in results:
        b = results["base"]
        for kind, r in results.items():
            if kind != "base":
                imp = sum(r[1][f] > b[1][f] for f in range(FOLDS))
                log(f"  {kind} vs base: {r[0] - b[0]:+.4f}; folds improved {imp}/5; deltas " + " ".join(f"{r[1][f] - b[1][f]:+.4f}" for f in range(FOLDS)))
    for kind in [w for w in a.write.split(",") if w]:
        if kind == "base":
            Xall, Xte = X, Xt
        else:
            Fall = np.zeros((len(X), trans_feats(CX[:2], *TM(0), kind).shape[1]), np.float32)
            for j in allk:
                m = Fr == j; Fall[m] = trans_feats(CX[m], *TM(j), kind)
            Xall = np.concatenate([X, Fall], 1); Xte = np.concatenate([Xt, trans_feats(CXt, *tm[()], kind)], 1)
        mdl = lgb.train(prm, lgb.Dataset(Xall, T), REF_ROUNDS)
        ref_t, nft = refiner_flips(fin_t, tlt, ott, mdl.predict(Xte), len(Lt), a.thr)
        out = os.path.join(r"E:\Claude code\wear\subs", f"sub_research_transition")
        write_sub(z["ids"], ref_t, out + ".csv")
        np.save(out + "_labo.npy", results[kind][2].astype(np.int8)); np.save(out + "_labt.npy", ref_t.astype(np.int8))
        k7 = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)["ref_test"]
        log(f"[{kind}] test: {nft} tiles flipped; agree K7 refined {np.mean(ref_t == k7):.4f}, agree fused pre-refiner {np.mean(ref_t == fin_t):.4f}; wrote {out}.csv")


if __name__ == "__main__":
    main()
