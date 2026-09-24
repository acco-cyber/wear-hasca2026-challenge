"""Graph purity: re-weight graph_smooth neighbour edges by a learned P(same label) (oracle graph purification = +0.024),
then purity2 soft chain decoding. python gpure.py eval|cv [gamma list]"""
import sys as _s, time
from lk import *
from purity2 import load_train_structs, edge_feats2, calib_soft, fit as fit_chain, TRAIN_SESS
import lightgbm as lgb

def graph_pairs(P, st, null_scale=0.5):
    n = len(P); Pn = P.copy(); Pn[:, 0] *= null_scale; Pn /= Pn.sum(1, keepdims=True)
    g = build_graph(st["cand"], st["lo"], n, k=10); Pg = graph_smooth(Pn, g, alpha=0.5, iters=5)
    A = np.concatenate([np.full(len(i), a) for a, (i, w) in enumerate(g)]); J = np.concatenate([i for i, w in g]); Wt = np.concatenate([w for i, w in g])
    lg = lambda x: np.log(np.clip(x, 1e-9, 1)); m = 0.5 * (Pg[A] + Pg[J])
    js = 0.5 * (Pg[A] * (lg(Pg[A]) - lg(m))).sum(1) + 0.5 * (Pg[J] * (lg(Pg[J]) - lg(m))).sum(1)
    F = np.stack([np.log(Wt / (1 - Wt + 1e-9) + 1e-9), (Pn[A] * Pn[J]).sum(1), (Pg[A] * Pg[J]).sum(1), js, Pn[A, 0], Pn[J, 0], Pg[A, 0], Pg[J, 0],
                  Pn[A].max(1), Pn[J].max(1), (Pg[A].argmax(1) == Pg[J].argmax(1)).astype(float)], 1).astype(np.float32)
    return g, A, J, F, Pn

def fit_graph(STs, oof):
    Xs, ys = [], []
    for st in STs:
        P = sim_P(oof, st); g, A, J, F, Pn = graph_pairs(P, st); Xs.append(F); ys.append((st["y"][A] == st["y"][J]).astype(int))
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.5, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, n_jobs=3).fit(np.concatenate(Xs), np.concatenate(ys))

def decode_g(P, st, gm, cm, gamma, tau=0.3, scale=1.0, D=DEC_NEW):
    g, A, J, F, Pn = graph_pairs(P, st, D["null_scale"]); ps = gm.predict_proba(F)[:, 1]
    g2 = []; o = 0
    for (i, w) in g:
        k = len(i); g2.append((i, w * ps[o:o + k] ** gamma)); o += k
    Pg2 = graph_smooth(Pn, g2, alpha=0.5, iters=5)
    a, b, succ, Fc, Pg = edge_feats2(P, st); p = cm.predict_proba(Fc)[:, 1]      # chain purity from the standard features
    succ2 = succ.copy(); succ2[a[p < tau]] = -1
    ps_node = np.full(len(P), 0.8); ps_node[a] = np.clip(0.8 + (p - 0.88) * scale, 0.05, 0.97)
    return calib_soft(Pg2, chains_from_succ(succ2), ps_node, lo=D["lo_c"], hi=D["hi_c"])

if __name__ == "__main__":
    oof = blend_oof(0.2); TS = load_train_structs(); t0 = time.time()
    gammas = [float(x) for x in (_s.argv[2] if len(_s.argv) > 2 else "0,1,2,4").split(",")]
    if _s.argv[1] == "eval":
        gm = fit_graph([st for _, _, st in TS], oof); pickle.dump(gm, open(os.path.join(EXP, "gpure_model.pkl"), "wb"))
        cm = pickle.load(open(os.path.join(EXP, "purity2_model.pkl"), "rb")); print("fit", f"({time.time()-t0:.0f}s)", flush=True)
        S0 = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); res = {}
        for s in EVAL:
            st = S0[s]; P = sim_P(oof, st)
            for gmm in gammas: res.setdefault(gmm, []).append(f1(st["y"], decode_g(P, st, gm, cm, gmm)))
            print(s, {k: round(v[-1], 4) for k, v in res.items()}, f"({time.time()-t0:.0f}s)", flush=True)
        for k, v in res.items(): print("EVAL gamma", k, f"mean {np.mean(v):.4f} | " + " ".join(f"{x:.4f}" for x in v))
    else:
        folds = [TRAIN_SESS[0::2], TRAIN_SESS[1::2]]; res = {}
        for k in range(2):
            tr = [st for s, _, st in TS if s in folds[1 - k]]; gm = fit_graph(tr, oof); cm = fit_chain(tr, oof)
            for s, f, st in TS:
                if s not in folds[k] or not f.endswith("struct.pkl"): continue
                P = sim_P(oof, st)
                for gmm in gammas: res.setdefault(gmm, []).append(f1(st["y"], decode_g(P, st, gm, cm, gmm)))
                print(s, {kk: round(v[-1], 4) for kk, v in res.items()}, f"({time.time()-t0:.0f}s)", flush=True)
        for kk, v in res.items(): print("CV gamma", kk, f"mean {np.mean(v):.4f}")
