"""Decoder parameter sweep on a sim struct pickle.
python tune.py --struct sim_struct_v2.pkl [--grid small|full]
"""
import argparse, time, itertools
from lk import *

def sinkhorn_graph(cand, lo, n, k=10, iters=20, temp=1.0):
    """Soft doubly-stochastic assignment marginals on the candidate graph (with a 'no successor' slack)."""
    valid = cand >= 0; rows = np.repeat(np.arange(n)[:, None], cand.shape[1], 1)
    Kv = np.where(valid, np.exp(np.clip(lo / temp, -30, 30)), 0.0)
    slack_r = np.ones(n); slack_c = np.ones(n)
    u = np.ones(n); v = np.ones(n)
    cc = np.where(valid, cand, 0)
    for _ in range(iters):
        rs = (Kv * v[cc]).sum(1) + slack_r; u = 1.0 / rs
        cs = np.bincount(cc[valid], weights=(Kv * u[:, None])[valid], minlength=n) + slack_c; v = 1.0 / cs
    Pm = Kv * u[:, None] * v[cc]
    lo2 = np.where(valid, np.log(np.clip(Pm, 1e-9, 1 - 1e-9) / (1 - np.clip(Pm, 1e-9, 1 - 1e-9))), -50)
    return lo2

def run(S, oof, cfg, sessions=EVAL):
    out = []
    for s in sessions:
        st = S[s]; P = sim_P(oof, st); n = st["n"]
        lo = st["lo"]
        if cfg.get("sink"): lo_g = sinkhorn_graph(st["cand"], lo, n, temp=cfg["sink"])
        else: lo_g = lo
        g = build_graph(st["cand"], lo_g, n, k=cfg["k"], min_logodds=cfg.get("gmin", -8.0))
        lab = decode(P, st["cand"], lo, st["succ0"], st["sc"], st["Lm"], thr=cfg["thr"], alpha=cfg["alpha"], iters=cfg["iters"],
                     p_stay=cfg["ps"], graph=g, lo_c=cfg.get("lo_c", 60), hi_c=cfg.get("hi_c", 160), null_scale=cfg.get("ns", 1.0))
        out.append(f1(st["y"], lab))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--struct", required=True); ap.add_argument("--grid", default="small")
    a = ap.parse_args(); p = a.struct if os.path.isabs(a.struct) else os.path.join(EXP, a.struct)
    S = pickle.load(open(p, "rb")); oof = blend_oof(0.2); t0 = time.time()
    base = dict(thr=-6.0, k=10, alpha=0.5, iters=5, ps=0.8)
    cfgs = [base]
    if a.grid.startswith("new"):
        nb = dict(base, ns=0.5, lo_c=80, hi_c=250); cfgs = [nb]
        if a.grid == "newfull":
            for kv in [dict(thr=-3.0), dict(k=20), dict(alpha=0.3), dict(alpha=0.7), dict(ps=0.7), dict(ps=0.9), dict(iters=10)]:
                c = dict(nb); c.update(kv); cfgs.append(c)
    elif a.grid == "small":
        for kv in [dict(thr=-3.0), dict(thr=-1.0), dict(thr=0.0), dict(k=5), dict(k=20), dict(alpha=0.7), dict(alpha=0.7, iters=10), dict(ps=0.9), dict(ps=0.7),
                   dict(sink=1.0), dict(alpha=0.3)]:
            c = dict(base); c.update(kv); cfgs.append(c)
    elif a.grid == "sink":
        for kv in [dict(sink=1.0), dict(sink=1.0, alpha=0.7), dict(sink=1.0, k=5), dict(sink=0.5), dict(sink=2.0)]:
            c = dict(base); c.update(kv); cfgs.append(c)
    for c in cfgs:
        r = run(S, oof, c); print(c, "mean %.4f" % np.mean(r), " ".join("%.4f" % x for x in r), f"({time.time()-t0:.0f}s)", flush=True)
