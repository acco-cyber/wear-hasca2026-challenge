"""Which link errors matter? Oracle-purify chains and/or graph of the baseline struct (new decoder)."""
from lk import *
S0 = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); oof = blend_oof(0.2); res = {}
for s in EVAL:
    o = S0[s]; n = o["n"]; y = o["y"]; P = sim_P(oof, o)
    succ = cut(o["succ0"], o["sc"], o["Lm"], -6.0)
    succ_p = succ.copy(); e = succ_p >= 0; bad = np.zeros(n, bool); bad[e] = y[succ_p[e]] != y[e]; succ_p[bad] = -1
    g = build_graph(o["cand"], o["lo"], n, k=10)
    g_p = [(i[y[i] == y[a]], w[y[i] == y[a]]) for a, (i, w) in enumerate(g)]
    succ_true = np.append(np.arange(1, n), -1)
    D = DEC_NEW
    def dec(sc_, g_):
        Pn = P.copy(); Pn[:, 0] *= D["null_scale"]; Pn /= Pn.sum(1, keepdims=True)
        Pg = graph_smooth(Pn, g_, alpha=0.5, iters=5)
        lab, _ = calibrate_counts(Pg, chains_from_succ(sc_), lo=D["lo_c"], hi=D["hi_c"], p_stay=0.8); return f1(y, lab)
    for nm, (sc_, g_) in {"base": (succ, g), "pure_chains": (succ_p, g), "pure_graph": (succ, g_p), "pure_both": (succ_p, g_p), "true_chain_graph_base": (succ_true, g)}.items():
        res.setdefault(nm, []).append(dec(sc_, g_))
    print(s, {k: round(v[-1], 4) for k, v in res.items()}, flush=True)
for nm, v in res.items(): print(f"{nm:24s} mean {np.mean(v):.4f}")
