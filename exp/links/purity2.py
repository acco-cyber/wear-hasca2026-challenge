"""Purity v2: + chain-context change-point features (mean smoothed probs over w windows before/after the edge along the chain)
+ more limb draws for training; hard cut (tau) and soft per-edge p_stay Viterbi.
python purity2.py eval    -> fit on all train structs, tau sweep on eval sessions + 2-fold CV over train sessions
"""
import sys as _s, time, glob
from lk import *
from purity import edge_feats, PF, TRAIN_SESS
import lightgbm as lgb

NC = 19

def ctx_feats(Pg, succ, a, b, ws=(3, 8)):
    n = len(Pg); chains = chains_from_succ(succ); pos = np.zeros(n, int); cid = np.zeros(n, int); clen = np.zeros(n, int)
    for k, ch in enumerate(chains):
        ch = np.array(ch); pos[ch] = np.arange(len(ch)); cid[ch] = k; clen[ch] = len(ch)
    cums = []
    for ch in chains:
        c = np.zeros((len(ch) + 1, NC)); c[1:] = np.cumsum(Pg[ch], 0); cums.append(c)
    out = []
    lg = lambda x: np.log(np.clip(x, 1e-9, 1))
    for w in ws:
        A = np.zeros((len(a), NC)); B = np.zeros((len(a), NC))
        for j, (x, y_) in enumerate(zip(a, b)):
            c = cums[cid[x]]; t = pos[x]; T = clen[x]
            lo_ = max(0, t - w + 1); A[j] = (c[t + 1] - c[lo_]) / (t + 1 - lo_)
            hi_ = min(T, t + 1 + w); B[j] = (c[hi_] - c[t + 1]) / (hi_ - t - 1)
        m = 0.5 * (A + B); js = 0.5 * (A * (lg(A) - lg(m))).sum(1) + 0.5 * (B * (lg(B) - lg(m))).sum(1)
        out += [(A * B).sum(1), (A.argmax(1) == B.argmax(1)).astype(float), js, (Pg[a] * B).sum(1), (A * Pg[b]).sum(1), A[:, 0], B[:, 0]]
    t = pos[a]; T = clen[a]
    out += [np.log1p(T), np.log1p(np.minimum(t, T - 1 - t))]
    return np.stack(out, 1).astype(np.float32)

def edge_feats2(P, st):
    a, b, succ, F, Pg = edge_feats(P, st)
    return a, b, succ, np.concatenate([F, ctx_feats(Pg, succ, a, b)], 1), Pg

def viterbi_soft(P, chains, ps_node, base_ps=0.8):
    n = len(P); out = P.argmax(1).copy(); logE = np.log(np.clip(P, 1e-6, 1))
    for ch in chains:
        if len(ch) == 1: continue
        E = logE[ch]; T = len(ch); delta = E[0].copy(); back = np.zeros((T, NC), np.int64)
        for t in range(1, T):
            ps = ps_node[ch[t - 1]]
            logT = np.full((NC, NC), np.log((1 - ps) / (NC - 1))); np.fill_diagonal(logT, np.log(ps))
            m = delta[:, None] + logT; back[t] = m.argmax(0); delta = m.max(0) + E[t]
        lab = np.zeros(T, np.int64); lab[-1] = delta.argmax()
        for t in range(T - 1, 0, -1): lab[t - 1] = back[t, lab[t]]
        out[ch] = lab
    return out

def calib_soft(P, chains, ps_node, lo=80, hi=250, iters=40, step=0.25):
    n = len(P); bb = np.zeros(NC); logP = np.log(np.clip(P, 1e-6, 1))
    for it in range(iters):
        Pb = np.exp(logP + bb[None, :]); Pb /= Pb.sum(1, keepdims=True)
        lab = viterbi_soft(Pb, chains, ps_node); cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo)[0] + 1; over = np.where(cnt[1:] > hi)[0] + 1
        if len(under) == 0 and len(over) == 0: break
        bb[under] += step; bb[over] -= step
    return lab

def fit(STs, oof):
    Xs, ys = [], []
    for st in STs:
        P = sim_P(oof, st); a, b, succ, F, Pg = edge_feats2(P, st); y = st["y"]; Xs.append(F); ys.append((y[a] == y[b]).astype(int))
    X = np.concatenate(Xs); yy = np.concatenate(ys)
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, n_jobs=3).fit(X, yy)

def decode2(P, st, model, tau=0.5, mode="hard", D=DEC_NEW, return_parts=False):
    a, b, succ, F, Pg = edge_feats2(P, st); p = model.predict_proba(F)[:, 1]
    if mode == "hard":
        succ2 = succ.copy(); succ2[a[p < tau]] = -1
        lab, _ = calibrate_counts(Pg, chains_from_succ(succ2), lo=D["lo_c"], hi=D["hi_c"], p_stay=0.8)
    else:   # soft: per-edge p_stay from p_same, plus hard cut below tau
        succ2 = succ.copy(); succ2[a[p < tau]] = -1
        ps_node = np.full(len(P), 0.8); ps_node[a] = np.clip(0.8 + (p - 0.88) * float(mode.split("_")[1]), 0.05, 0.97)
        lab = calib_soft(Pg, chains_from_succ(succ2), ps_node, lo=D["lo_c"], hi=D["hi_c"])
    if return_parts: return lab, succ2, p
    return lab

def load_train_structs():
    out = []
    for f in sorted(glob.glob(os.path.join(EXP, "purity_train_struct*.pkl"))):
        d = pickle.load(open(f, "rb"))
        for s, st in d.items(): out.append((s, os.path.basename(f), dict(st, Lm=st["Lm"].astype(np.float32))))
    return out

if __name__ == "__main__":
    oof = blend_oof(0.2); TS = load_train_structs(); t0 = time.time()
    print("train structs:", len(TS), sorted(set(f for _, f, _ in TS)), flush=True)
    MODES = [("hard", 0.0), ("hard", 0.4), ("hard", 0.5), ("hard", 0.6), ("soft_1.0", 0.3), ("soft_1.0", 0.0)]
    if len(_s.argv) > 2: MODES = [(m.split(":")[0], float(m.split(":")[1])) for m in _s.argv[2].split(",")]
    if _s.argv[1] == "eval":
        mp_ = os.path.join(EXP, "purity2_model.pkl")
        if os.path.exists(mp_) and len(_s.argv) > 2: model = pickle.load(open(mp_, "rb"))
        else: model = fit([st for _, _, st in TS], oof); pickle.dump(model, open(mp_, "wb"))
        print("fit done", f"({time.time()-t0:.0f}s)", flush=True)
        S0 = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); res = {}
        for s in EVAL:
            st = S0[s]; P = sim_P(oof, st); y = st["y"]
            for md, tau in MODES: res.setdefault((md, tau), []).append(f1(y, decode2(P, st, model, tau, md)))
            print(s, {k: round(v[-1], 4) for k, v in res.items()}, f"({time.time()-t0:.0f}s)", flush=True)
        for k, v in res.items(): print("EVAL", k, f"mean {np.mean(v):.4f} | " + " ".join(f"{x:.4f}" for x in v))
    elif _s.argv[1] == "cv":
        folds = [TRAIN_SESS[0::2], TRAIN_SESS[1::2]]; res = {}
        for k in range(2):
            m = fit([st for s, _, st in TS if s in folds[1 - k]], oof)
            for s, f, st in TS:
                if s not in folds[k] or not f.endswith("struct.pkl"): continue
                P = sim_P(oof, st); y = st["y"]
                for md, tau in MODES: res.setdefault((md, tau), []).append(f1(y, decode2(P, st, m, tau, md)))
                print(s, {kk: round(v[-1], 4) for kk, v in res.items()}, f"({time.time()-t0:.0f}s)", flush=True)
        for kk, v in res.items(): print("CV", kk, f"mean {np.mean(v):.4f}")
