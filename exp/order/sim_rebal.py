"""Targeted within-family re-balancing (protocol: variants of a family have ~equal durations within a subject).
After the e7 decode, for each family whose decoded variant counts are imbalanced (max/min >= trig), freeze the family's
window set, restrict those windows to the family's variants, and re-decode with per-variant bounds [a*m, b*m]
(m = family mean count). python sim_rebal.py cfgs.json tag"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olib import SESS, ALL, HERE, NC, FAMS, to_P
_G = {}
NEG = -30.0

def rebal(L, lab, st, c, pre=None):
    from mydec import decode, prep
    pre = pre or prep(st, len(lab))
    cnt = np.bincount(lab, minlength=NC).astype(float); L2 = L.copy(); lo = np.full(NC - 1, 80.0); hi = np.full(NC - 1, 250.0)
    fams = [f for f in c.get("fams", ["push", "sit", "lun"])]; trig = []
    for f in fams:
        cls = FAMS[f]; v = cnt[cls]
        if v.min() <= 0 or v.max() / v.min() < c["trig"]: continue
        trig.append(f); m = v.mean(); inf = np.isin(lab, cls)
        mask_out = np.ones(NC, bool); mask_out[cls] = False
        L2[np.ix_(inf, mask_out)] = NEG; L2[np.ix_(~inf, np.array(cls))] = NEG
        for k in cls: lo[k - 1] = c["a"] * m; hi[k - 1] = c["b"] * m
    if not trig: return lab, trig
    return decode(to_P(L2), st, pre, lo=lo, hi=hi, iters=c.get("iters", 40)), trig

def _init():
    sys.path.insert(0, HERE)
    from tlib import load_structs
    S = {}
    for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
    _G.update(S=S, C=pickle.load(open(os.path.join(HERE, "cache_e7.pkl"), "rb")))

def _job(args):
    s, cfgs = args
    from sklearn.metrics import f1_score
    from mydec import prep
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; L1 = d["L1"].astype(np.float64); pre = prep(st, len(y))
    out = {"e7": f1_score(y, d["lab1"], average="macro")}; labs = {}; trigs = {}
    for c in cfgs:
        lab, trig = rebal(L1, d["lab1"], st, c, pre); k = json.dumps(c, sort_keys=True)
        out[k] = f1_score(y, lab, average="macro"); labs[k] = lab; trigs[k] = trig
    return s, out, labs, trigs

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); tag = sys.argv[2]; t0 = time.time(); R = {}; LB = {}; TR = {}
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for s, out, labs, trigs in ex.map(_job, [(s, cfgs) for s in ALL]):
            R[s] = out; LB[s] = labs; TR[s] = trigs
            print(s, f"{time.time()-t0:.0f}s", " ".join(f"[{','.join(t)}] {out[k]-out['e7']:+.4f}" for k, t in trigs.items()), flush=True)
    lines = []
    for k in R[ALL[0]]:
        means = {w: np.mean([R[s][k] for s in ss]) for w, ss in SESS.items()}; a18 = np.mean([R[s][k] for s in ALL])
        up = sum(R[s][k] > R[s]["e7"] + 1e-9 for s in ALL); dn = sum(R[s][k] < R[s]["e7"] - 1e-9 for s in ALL)
        lines.append(f"{k:75s} " + " ".join(f"{w} {v:.4f}" for w, v in means.items()) + f" | all18 {a18:.4f} | up {up} down {dn}")
    print("\n".join(lines))
    with open(os.path.join(HERE, f"res_{tag}.txt"), "w") as f: f.write("\n".join(lines) + "\n")
    pickle.dump(dict(R=R, labs=LB, trig=TR), open(os.path.join(HERE, f"res_{tag}.pkl"), "wb"))
