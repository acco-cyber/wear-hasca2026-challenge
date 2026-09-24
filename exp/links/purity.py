"""Edge-purity model: predict whether an assignment edge a->succ(a) joins two windows of the SAME label, using link scores +
(graph-smoothed) class probabilities; cut edges with low predicted purity before the chain Viterbi.
Oracle: cutting all cross-label chain edges lifts sim 0.7543 -> 0.8140 (new decoder).
python purity.py build     -> honest structures (old scorer, fit on OLD_FIT) for 10 other non-eval sessions (cached)
python purity.py fit       -> fit purity LightGBM on them, sweep tau on eval sessions
"""
import sys as _s, time
from lk import *
from chain import make_training_pairs, assignment as assign_old   # src/chain.py (read-only)
import lightgbm as lgb

TRAIN_SESS = ["sbj_11", "sbj_13", "sbj_15", "sbj_17", "sbj_18", "sbj_2", "sbj_4", "sbj_6", "sbj_8", "sbj_9"]
PF = ["sc", "rowmax2", "colmax2", "rev", "dotP", "dotPg", "eqP", "eqPg", "maxPa", "maxPb", "nullPa", "nullPb", "nullPga", "nullPgb",
      "jsPg", "ctx_prev", "ctx_next", "dotPg2", "rank_sc"]

def smooth(P, cand, lo, n, null_scale=0.5):
    Pn = P.copy(); Pn[:, 0] *= null_scale; Pn /= Pn.sum(1, keepdims=True)
    return Pn, graph_smooth(Pn, build_graph(cand, lo, n, k=10), alpha=0.5, iters=5)

def edge_feats(P, st, null_scale=0.5):
    n = len(P); Lm = st["Lm"].astype(np.float32)
    succ = cut(st["succ0"], st["sc"], Lm, -6.0); a = np.where(succ >= 0)[0]; b = succ[a]
    Pn, Pg = smooth(P, st["cand"], st["lo"], n, null_scale)
    srt = np.sort(Lm[a], 1); rowmax2 = st["sc"][a] - srt[:, -2]
    colv = Lm[:, b].T; srtc = np.sort(colv, 1); colmax2 = st["sc"][a] - srtc[:, -2]
    rank_sc = (Lm[a] > st["sc"][a][:, None]).sum(1)
    rev = Lm[b, a]
    dotP = (Pn[a] * Pn[b]).sum(1); dotPg = (Pg[a] * Pg[b]).sum(1)
    eqP = (Pn[a].argmax(1) == Pn[b].argmax(1)).astype(np.float32); eqPg = (Pg[a].argmax(1) == Pg[b].argmax(1)).astype(np.float32)
    m = 0.5 * (Pg[a] + Pg[b]); lg = lambda x: np.log(np.clip(x, 1e-9, 1))
    js = 0.5 * (Pg[a] * (lg(Pg[a]) - lg(m))).sum(1) + 0.5 * (Pg[b] * (lg(Pg[b]) - lg(m))).sum(1)
    pred = np.full(n, -1); pred[b] = a
    pa = pred[a]; ctx_prev = np.where(pa >= 0, (Pg[np.maximum(pa, 0)] * Pg[b]).sum(1), np.nan)
    sb = succ[b]; ctx_next = np.where(sb >= 0, (Pg[a] * Pg[np.maximum(sb, 0)]).sum(1), np.nan)
    # 2-step smoothed agreement along the chain
    Pc = Pg.copy(); Pc[a] += Pg[b]; Pc[b] += Pg[a]; Pc /= Pc.sum(1, keepdims=True); dotPg2 = (Pc[a] * Pc[b]).sum(1)
    F = np.stack([st["sc"][a], rowmax2, colmax2, rev, dotP, dotPg, eqP, eqPg, Pn[a].max(1), Pn[b].max(1), Pn[a, 0], Pn[b, 0], Pg[a, 0], Pg[b, 0],
                  js, ctx_prev, ctx_next, dotPg2, np.log1p(rank_sc)], 1).astype(np.float32)
    return a, b, succ, F, Pg

def build(seed=7, fname="purity_train_struct.pkl"):
    meta, imu, vid = load_prep(); sl = session_slices(meta); scorer = pickle.load(open(os.path.join(WORK, "scorer.pkl"), "rb"))
    rng = np.random.RandomState(seed); out = {}; t0 = time.time()
    for s in TRAIN_SESS:
        a_, b_ = sl[s]; n = b_ - a_
        F, L, cand, limb, Xi = make_training_pairs(np.asarray(vid[a_:b_], np.float32), np.asarray(imu[a_:b_], np.float32), rng)
        lo = scorer.logodds(F); succ0, scs, Lm = assign_old(cand, lo, n)
        out[s] = dict(a=a_, b=b_, n=n, y=meta.y.to_numpy()[a_:b_], limb=limb, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm.astype(np.float16))
        print("built", s, f"cand_rec {L[:-1].any(1).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(out, open(os.path.join(EXP, fname), "wb"))

def fit_model(ST, oof):
    Xs, ys = [], []
    for s, st in ST.items():
        P = sim_P(oof, st); a, b, succ, F, Pg = edge_feats(P, st); y = st["y"]
        Xs.append(F); ys.append((y[a] == y[b]).astype(int))
    X = np.concatenate(Xs); yy = np.concatenate(ys)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1, n_jobs=3).fit(X, yy)
    print("purity fit: edges", len(yy), "same rate", yy.mean().round(4), "imp", sorted(zip(PF, m.feature_importances_), key=lambda x: -x[1])[:10], flush=True)
    return m

def decode_pure(P, st, model, tau, D=DEC_NEW, return_parts=False):
    n = len(P); a, b, succ, F, Pg = edge_feats(P, st, D["null_scale"])
    p = model.predict_proba(F)[:, 1]; succ2 = succ.copy(); succ2[a[p < tau]] = -1
    lab, _ = calibrate_counts(Pg, chains_from_succ(succ2), lo=D["lo_c"], hi=D["hi_c"], p_stay=0.8)
    if return_parts: return lab, succ2, p
    return lab

if __name__ == "__main__":
    mode = _s.argv[1]
    if mode == "build": build()
    elif mode == "build2": build(seed=int(_s.argv[2]), fname=f"purity_train_struct_s{_s.argv[2]}.pkl")
    elif mode == "fit":
        oof = blend_oof(0.2); ST = pickle.load(open(os.path.join(EXP, "purity_train_struct.pkl"), "rb"))
        model = fit_model(ST, oof); pickle.dump(model, open(os.path.join(EXP, "purity_model.pkl"), "wb"))
        S0 = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); res = {}; t0 = time.time()
        for s in EVAL:
            st = S0[s]; P = sim_P(oof, st); y = st["y"]
            for tau in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8):
                lab, succ2, p = decode_pure(P, st, model, tau, return_parts=True)
                e = succ2 >= 0; res.setdefault(tau, []).append((f1(y, lab), float((y[e] == y[succ2[e]]).mean()), int(e.sum())))
            print(s, {t: round(v[-1][0], 4) for t, v in res.items()}, f"({time.time()-t0:.0f}s)", flush=True)
        for t, v in res.items():
            v = np.array(v); print(f"tau {t}: mean F1 {v[:,0].mean():.4f} | " + " ".join(f"{x:.4f}" for x in v[:, 0]) + f" | same-label edges {v[:,1].mean():.4f} edges {v[:,2].mean():.0f}")
