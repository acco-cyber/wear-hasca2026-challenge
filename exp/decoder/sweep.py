"""Staged sweeps with the numba decoder. python sweep.py A|B|C"""
import sys, time, itertools, numpy as np, pandas as pd
from common import *
from nbvit import pack, calib, logT_matrix, viterbi
from harness import data

oof, S = data()
SESS = [s for s in EVAL + EXTRA if s in S]
_sm = {}; _pk = {}
def Pg(s, P, k=10, alpha=0.5, iters=5):
    key = (s, k, alpha, iters)
    if key not in _sm:
        st = S[s]; _sm[key] = np.log(np.clip(smooth(P, graph_matrix(st["cand"], st["lo"], st["n"], k=k), alpha, iters), 1e-6, 1)) if alpha > 0 else np.log(np.clip(P, 1e-6, 1))
    return _sm[key]
def packed(s, thr=-6.0):
    if (s, thr) not in _pk: _pk[(s, thr)] = pack(base_chains(S[s], thr))
    return _pk[(s, thr)]
def edge_w(s, thr, beta, center=0.0):
    """per-position switch-cost multiplier from the link log-odds of the incoming chain edge"""
    order, starts, lens = packed(s, thr); st = S[s]; sc = st["sc"]
    prev = np.r_[-1, order[:-1]]; is_start = np.zeros(len(order), bool); is_start[starts] = True
    e = np.where(is_start, 0.0, sc[np.maximum(prev, 0)])
    return np.where(is_start, 1.0, 2 / (1 + np.exp(-beta * (e - center))))

def evaluate(cfgs, name):
    rows = []
    for s in SESS:
        st = S[s]; P = sess_P(oof, st); y = st["y"]
        for cname, cfg in cfgs.items():
            c = dict(k=10, alpha=0.5, iters=5, thr=-6.0, lo=60, hi=160, ps=0.8, temp=1.0, null_switch=None, beta=None, step=0.25, citers=40, center=0.0)
            c.update(cfg)
            L = Pg(s, P, c["k"], c["alpha"], c["iters"]) / c["temp"]
            pk = packed(s, c["thr"]); ew = edge_w(s, c["thr"], c["beta"], c["center"]) if c["beta"] else None
            T = logT_matrix(c["ps"], null_switch=c["null_switch"])
            lab, b = calib(L, pk, c["lo"], c["hi"], iters=c["citers"], step=c["step"], ew=ew, logT=T)
            rows.append(dict(session=s, cfg=cname, f1=mf1(y, lab)))
        print(s, "done", flush=True)
    df = pd.DataFrame(rows).pivot(index="cfg", columns="session", values="f1")
    df["EVAL"] = df[[s for s in EVAL]].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1); df["ALL"] = df[SESS].mean(1)
    df = df.sort_values("ALL", ascending=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    print(name); print(df.round(4).to_string(), flush=True); df.to_csv(f"sweep_{name}.csv")
    return df

from vidknn import mp_desc, knn_edges, train_vid
_vg = {}
def vid_graph_logP(s, P, kv=5, wv=0.5, k=10, alpha=0.5, iters=5, sym=True, simw=False):
    key = (s, kv, wv, k, alpha, iters, sym, simw)
    if key not in _vg:
        st = S[s]; n = st["n"]; D = mp_desc(train_vid()[st["a"]:st["b"]]); r, c, w = knn_edges(D, kv)
        w = (w if simw else np.ones_like(w)) * wv
        if sym: r, c, w = np.r_[r, c], np.r_[c, r], np.r_[w, w]
        W = graph_matrix(st["cand"], st["lo"], n, k=k, extra=(r, c, w)) if k > 0 else sp.csr_matrix((w, (r, c)), shape=(n, n))
        _vg[key] = np.log(np.clip(smooth(P, W, alpha, iters), 1e-6, 1))
    return _vg[key]

def evaluate_v(cfgs, name):
    rows = []
    for s in SESS:
        st = S[s]; P = sess_P(oof, st); y = st["y"]
        for cname, cfg in cfgs.items():
            c = dict(kv=5, wv=0.5, k=10, alpha=0.5, iters=5, sym=True, simw=False, lo=85, hi=160, ps=0.8)
            c.update(cfg)
            L = vid_graph_logP(s, P, c["kv"], c["wv"], c["k"], c["alpha"], c["iters"], c["sym"], c["simw"])
            lab, b = calib(L, packed(s, -6.0), c["lo"], c["hi"], p_stay=c["ps"])
            rows.append(dict(session=s, cfg=cname, f1=mf1(y, lab), f1_argmax=mf1(y, L.argmax(1))))
        print(s, "done", flush=True)
    D = pd.DataFrame(rows)
    for col in ("f1", "f1_argmax"):
        df = D.pivot(index="cfg", columns="session", values=col)
        df["EVAL"] = df[[s for s in EVAL]].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1); df["ALL"] = df[SESS].mean(1)
        df = df.sort_values("ALL", ascending=False)
        pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
        print(name, col); print(df.round(4).to_string(), flush=True); df.to_csv(f"sweep_{name}_{col}.csv")

from mrf import calib_mrf
def rownorm(W):
    d = np.asarray(W.sum(1)).ravel(); return (sp.diags(1 / np.where(d > 0, d, 1)) @ W).tocsr()
_wg = {}
def mrf_W(s, kv, wv, k=10):
    key = (s, kv, wv, k)
    if key not in _wg:
        st = S[s]; n = st["n"]; W = graph_matrix(st["cand"], st["lo"], n, k=k)
        if wv > 0:
            D = mp_desc(train_vid()[st["a"]:st["b"]]); r, c, w = knn_edges(D, kv)
            W = W + sp.csr_matrix((np.full(len(r), wv), (np.r_[r], np.r_[c])), shape=(n, n))
        W = W + W.T; _wg[key] = rownorm(W)
    return _wg[key]

_ns = {}
def Pg_ns(s, P, ns, k=10, alpha=0.5, iters=5):
    if ns == 1.0: return Pg(s, P, k, alpha, iters)
    key = (s, ns, k, alpha, iters)
    if key not in _ns:
        Q = P.copy(); Q[:, 0] *= ns; Q /= Q.sum(1, keepdims=True); st = S[s]
        _ns[key] = np.log(np.clip(smooth(Q, graph_matrix(st["cand"], st["lo"], st["n"], k=k), alpha, iters), 1e-6, 1))
    return _ns[key]

def evaluate_m(cfgs, name):
    rows = []
    for s in SESS:
        st = S[s]; P = sess_P(oof, st); y = st["y"]
        for cname, cfg in cfgs.items():
            c = dict(kv=5, wv=0.0, gkv=5, gwv=0.0, lam=1.0, icm_iters=10, lo=85, hi=160, ps=0.8, mk=10, ns=1.0, citers=40, alpha=0.5)
            c.update(cfg)
            L = vid_graph_logP(s, P, c["gkv"], c["gwv"], 10, 0.5, 5, True, False) if c["gwv"] > 0 else Pg_ns(s, P, c["ns"], alpha=c["alpha"])
            lab, b = calib_mrf(L, packed(s, -6.0), mrf_W(s, c["kv"], c["wv"], c["mk"]), c["lo"], c["hi"], c["ps"], c["lam"], c["icm_iters"], iters=c["citers"])
            rows.append(dict(session=s, cfg=cname, f1=mf1(y, lab)))
        print(s, "done", flush=True)
    df = pd.DataFrame(rows).pivot(index="cfg", columns="session", values="f1")
    df["EVAL"] = df[[s for s in EVAL]].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1); df["ALL"] = df[SESS].mean(1)
    df = df.sort_values("ALL", ascending=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    print(name); print(df.round(4).to_string(), flush=True); df.to_csv(f"sweep_{name}.csv")

from mrf2 import calib_mrf2
_wp = {}
def mrf_Wp(s, power=1.0, k=10):
    key = (s, power, k)
    if key not in _wp:
        st = S[s]; n = st["n"]; W = graph_matrix(st["cand"], st["lo"], n, k=k)
        if power != 1.0: W = W.power(power)
        W = W + W.T; _wp[key] = rownorm(W)
    return _wp[key]

def evaluate_m2(cfgs, name):
    rows = []
    for s in SESS:
        st = S[s]; P = sess_P(oof, st); y = st["y"]
        for cname, cfg in cfgs.items():
            c = dict(lam=4.0, lam_null=None, icm_iters=10, lo=80, hi=250, ps=0.7, ns=0.5, frac=0.5, seed=0, polish=0, power=1.0, vote=1, alpha=0.5)
            c.update(cfg)
            L = Pg_ns(s, P, c["ns"], alpha=c["alpha"]); W = mrf_Wp(s, c["power"])
            if "ens" in c:
                labs = [calib_mrf2(L, packed(s, -6.0), mrf_Wp(s, pw), c["lo"], c["hi"], c["ps"], c["lam"], None, c["icm_iters"], seed=sd)[0]
                        for pw, sd in c["ens"]]
            else:
                labs = [calib_mrf2(L, packed(s, -6.0), W, c["lo"], c["hi"], c["ps"], c["lam"], c["lam_null"], c["icm_iters"],
                               frac=c["frac"], seed=c["seed"] + v, polish=c["polish"])[0] for v in range(c["vote"])]
            if len(labs) == 1: lab = labs[0]
            else:
                V = np.zeros((len(y), NC)); [np.add.at(V, (np.arange(len(y)), l), 1) for l in labs]; lab = V.argmax(1)
            rows.append(dict(session=s, cfg=cname, f1=mf1(y, lab)))
        print(s, "done", flush=True)
    df = pd.DataFrame(rows).pivot(index="cfg", columns="session", values="f1")
    df["EVAL"] = df[[s for s in EVAL]].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1); df["ALL"] = df[SESS].mean(1)
    df = df.sort_values("ALL", ascending=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    print(name); print(df.round(4).to_string(), flush=True); df.to_csv(f"sweep_{name}.csv")

stage = sys.argv[1]
if stage == "L":
    cfgs = {"mix6_p1_p.5": dict(ens=[(1.0, 0), (1.0, 1), (1.0, 2), (0.5, 0), (0.5, 1), (0.5, 2)]),
            "mix3_p1": dict(ens=[(1.0, 0), (1.0, 1), (1.0, 2)]), "mix3_p.5": dict(ens=[(0.5, 0), (0.5, 1), (0.5, 2)])}
    evaluate_m2(cfgs, "L")
if stage == "K":
    cfgs = {"mrf4": {}, "pow0.5": dict(power=0.5), "pow0.25": dict(power=0.25), "pow0.01": dict(power=0.01),
            "pow0.5_s1": dict(power=0.5, seed=1), "pow0.5_s2": dict(power=0.5, seed=2), "pow0.5_vote5": dict(power=0.5, vote=5),
            "pow0.5_lam3": dict(power=0.5, lam=3.0), "pow0.5_lam5": dict(power=0.5, lam=5.0)}
    evaluate_m2(cfgs, "K")
if stage == "K2":
    cfgs = {"pow0.25_s1": dict(power=0.25, seed=1), "pow0.25_s2": dict(power=0.25, seed=2), "pow0.25_vote5": dict(power=0.25, vote=5),
            "pow0.01_s1": dict(power=0.01, seed=1), "pow0.01_vote5": dict(power=0.01, vote=5), "pow0.25_polish5": dict(power=0.25, polish=5)}
    evaluate_m2(cfgs, "K2")
if stage == "I":
    cfgs = {"mrf4": {}, "seed1": dict(seed=1), "seed2": dict(seed=2), "vote5": dict(vote=5),
            "lamnull2": dict(lam_null=2.0), "lamnull3": dict(lam_null=3.0), "lamnull6": dict(lam_null=6.0),
            "hi160": dict(hi=160), "hi200": dict(hi=200)}
    evaluate_m2(cfgs, "I")
if stage == "J":
    cfgs = {"mrf4": {}, "frac0.3": dict(frac=0.3), "frac1": dict(frac=1.0), "polish5": dict(polish=5), "pow2": dict(power=2.0),
            "pow0.5": dict(power=0.5), "icm5": dict(icm_iters=5), "alpha0.3": dict(alpha=0.3), "alpha0.7": dict(alpha=0.7)}
    evaluate_m2(cfgs, "J")
if stage == "H":
    b = dict(lam=4.0, ns=0.5, lo=80, hi=250, ps=0.7)
    cfgs = {"best_G": b, "ns0.35": dict(b, ns=0.35), "ns0.25": dict(b, ns=0.25), "lo75": dict(b, lo=75), "lo85": dict(b, lo=85),
            "lam5": dict(b, lam=5.0), "ps.6": dict(b, ps=0.6), "icm20": dict(b, icm_iters=20)}
    evaluate_m(cfgs, "H")
if stage == "G":
    cfgs = {}
    for ns in (0.7, 0.5):
        cfgs[f"lam4_ns{ns}_80_250"] = dict(lam=4.0, ns=ns, lo=80, hi=250)
        cfgs[f"lam4_ns{ns}_80_250_ps.7"] = dict(lam=4.0, ns=ns, lo=80, hi=250, ps=0.7)
        cfgs[f"lam3_ns{ns}_80_250_ci80"] = dict(lam=3.0, ns=ns, lo=80, hi=250, citers=80)
    cfgs["lam3_ns0.7_85_250_a0.3"] = dict(lam=3.0, ns=0.7, lo=85, hi=250, alpha=0.3)
    evaluate_m(cfgs, "G")
if stage == "F":
    cfgs = {"lead_ns0.5_80_250_nomrf": dict(lam=0.0, ns=0.5, lo=80, hi=250)}
    lams = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3.0, 6.0]
    for lam in lams:
        for ns in (1.0, 0.7, 0.5):
            for lo, hi in ((85, 160), (80, 250), (90, 250)):
                cfgs[f"lam{lam}_ns{ns}_{lo}_{hi}"] = dict(lam=lam, ns=ns, lo=lo, hi=hi)
    evaluate_m(cfgs, "F_" + "_".join(str(l) for l in lams))
if stage == "E":
    cfgs = {}
    for lam in (3.0, 4.0, 6.0, 9.0): cfgs[f"lam{lam}"] = dict(lam=lam)
    for lam in (4.0, 6.0): cfgs[f"lam{lam}_it30"] = dict(lam=lam, icm_iters=30)
    for lam in (4.0, 6.0): cfgs[f"lam{lam}_mk20"] = dict(lam=lam, mk=20)
    for lam in (4.0, 6.0): cfgs[f"lam{lam}_mk5"] = dict(lam=lam, mk=5)
    for lam in (4.0, 6.0): cfgs[f"lam{lam}_hi200_ps.7"] = dict(lam=lam, hi=200, ps=0.7)
    for lam in (4.0, 6.0): cfgs[f"lam{lam}_v10w0.6"] = dict(lam=lam, kv=10, wv=0.6)
    evaluate_m(cfgs, "E")
if stage == "D":
    cfgs = {"nomrf": dict(lam=0.0)}
    for lam in (0.5, 1.0, 2.0, 3.0): cfgs[f"lam{lam}"] = dict(lam=lam)
    for lam in (1.0, 2.0): cfgs[f"lam{lam}_v5w0.5"] = dict(lam=lam, kv=5, wv=0.5)
    for lam in (1.0, 2.0): cfgs[f"lam{lam}_v10w1"] = dict(lam=lam, kv=10, wv=1.0)
    evaluate_m(cfgs, "D")
if stage == "C":
    cfgs = {"link_only": dict(wv=0.0, kv=1)}
    for kv, wv in itertools.product((5, 10), (0.3, 0.6, 1.0)):
        cfgs[f"kv{kv}_wv{wv}"] = dict(kv=kv, wv=wv)
    cfgs["kv5_wv0.6_a0.7_i10"] = dict(kv=5, wv=0.6, alpha=0.7, iters=10)
    cfgs["kv10_wv1.0_a0.7_i10"] = dict(kv=10, wv=1.0, alpha=0.7, iters=10)
    cfgs["kv5_wv0.6_simw"] = dict(kv=5, wv=0.6, simw=True)
    cfgs["vidonly_kv10"] = dict(kv=10, wv=1.0, k=0)
    evaluate_v(cfgs, "C")
if stage == "A":
    cfgs = {"base": {}}
    for (k, a, it), (lo, hi), ps in itertools.product([(10, 0.5, 5), (10, 0.7, 10), (20, 0.6, 8), (5, 0.5, 5), (10, 0.0, 0)],
                                                       [(75, 160), (85, 160), (85, 200), (90, 180)], [0.7, 0.8, 0.9]):
        cfgs[f"k{k}a{a}i{it}_{lo}_{hi}_ps{ps}"] = dict(k=k, alpha=a, iters=it, lo=lo, hi=hi, ps=ps)
    evaluate(cfgs, "A")
elif stage == "B":
    base = dict(lo=85, hi=160, ps=0.8)
    cfgs = {"b85": base}
    for thr in (-8.0, -4.0, -2.0): cfgs[f"thr{thr}"] = dict(base, thr=thr)
    for ns in (0.05, 0.1): cfgs[f"ns{ns}"] = dict(base, null_switch=ns)
    for ns in (0.05, 0.1): cfgs[f"ns{ns}_ps.9"] = dict(base, null_switch=ns, ps=0.9)
    for beta in (0.3, 0.6, 1.0): cfgs[f"beta{beta}"] = dict(base, beta=beta)
    for beta, cen in ((0.5, -3.0), (0.5, 2.0)): cfgs[f"beta{beta}c{cen}"] = dict(base, beta=beta, center=cen)
    for t in (0.7, 1.5): cfgs[f"temp{t}"] = dict(base, temp=t)
    cfgs["step.1_it100"] = dict(base, step=0.1, citers=100)
    cfgs["step.5_it40"] = dict(base, step=0.5, citers=40)
    evaluate(cfgs, "B")
