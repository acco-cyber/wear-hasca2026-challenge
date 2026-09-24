"""Struct variants from per-session successor log-odds (lo) and same-label log-odds (lq).
Variant spec "name:a,b,tau" -> graph logodds = a*lo + b*lq ; chain edges with lq(i,succ0[i]) < tau are cut (tau=-99 none).
python build_variants.py <pred_tag> "bS:1,0,-99;bQ:0,1,-99;..." """
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
FD = os.path.join(XD, "feats"); PD = os.path.join(XD, "preds"); AD = os.path.join(XD, "assign"); os.makedirs(AD, exist_ok=True)

def _assign(args):
    tag, s = args; p = os.path.join(AD, f"{tag}_{s}.npz")
    if os.path.exists(p): return s
    d = np.load(os.path.join(FD, f"{s}_0.npz")); pr = np.load(os.path.join(PD, f"{tag}_{s}.npz"))
    succ0, sc, Lm = assignment(d["cand"], pr["lo"], len(d["y"])); np.savez(p, succ0=succ0, sc=sc, Lm=Lm.astype(np.float16)); return s

def edge_lq(cand, lq, succ):
    n = len(succ); out = np.full(n, -50.0, np.float32)
    for i in range(n):
        j = succ[i]; k = np.where(cand[i] == j)[0]
        if len(k): out[i] = lq[i, k[0]]
    return out

if __name__ == "__main__":
    tag = sys.argv[1]; specs = [x.split(":") for x in sys.argv[2].split(";")]
    allS = [s for w in SESS for s in SESS[w]]
    with ProcessPoolExecutor(3) as ex: list(ex.map(_assign, [(tag, s) for s in allS]))
    olds = {}
    for w in SESS:
        olds.update(pickle.load(open(os.path.join(WORK, "sim_struct.pkl") if w == "eval" else os.path.join(TD, f"{w}_struct.pkl"), "rb")))
    for name, par in specs:
        a_, b_, tau = [float(x) for x in par.split(",")]
        for w in SESS:
            out = {}
            for s in SESS[w]:
                d = np.load(os.path.join(FD, f"{s}_0.npz")); pr = np.load(os.path.join(PD, f"{tag}_{s}.npz")); asg = np.load(os.path.join(AD, f"{tag}_{s}.npz"))
                cand = d["cand"]; lo, lq = pr["lo"], pr["lq"]; o = olds[s]
                g = np.where(cand >= 0, a_ * lo + b_ * lq, -50.0).astype(np.float32)
                sc = asg["sc"].copy(); succ0 = asg["succ0"]
                if tau > -90:
                    e = edge_lq(cand, lq, succ0); sc = np.where(e < tau, np.minimum(sc, -50.0), sc).astype(np.float32)
                out[s] = dict(a=o["a"], b=o["b"], n=o["n"], y=o["y"], limb=np.asarray(o["limb"]), cand=cand, lo=g, succ0=succ0, sc=sc, Lm=asg["Lm"])
            pickle.dump(out, open(os.path.join(XD, f"struct_{name}_{w}.pkl"), "wb"))
        print("built", name, flush=True)
