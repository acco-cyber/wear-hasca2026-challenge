"""Sim of the bout-group bisection variant assignment on top of e7. python sim_bisect.py cfgs.json tag"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olib import SESS, ALL, HERE, NC, to_P
_G = {}

def _init():
    sys.path.insert(0, HERE)
    from tlib import load_structs
    S = {}
    for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
    _G.update(S=S, C=pickle.load(open(os.path.join(HERE, "cache_e7.pkl"), "rb")))

def _job(args):
    s, cfgs = args
    from sklearn.metrics import f1_score
    from mydec import decode, prep
    from bisect_core import apply, _W
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; L1 = d["L1"].astype(np.float64); n = len(y); pre = prep(st, n); W = _W(st, n)
    out = {"e7": f1_score(y, d["lab1"], average="macro")}; labs = {}
    for c in cfgs:
        lab2 = d["lab1"]
        for r in range(c.get("rounds", 1)):
            lab2, L2, info = apply(L1, lab2, st, c, W)
            if c["mode"] == "soft": lab2 = decode(to_P(L2), st, pre)
        if c.get("post_hard"): lab2, _, _ = apply(L2, lab2, st, dict(c, mode="hard"), W)
        k = json.dumps(c, sort_keys=True); out[k] = f1_score(y, lab2, average="macro"); labs[k] = lab2
    return s, out, labs

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); tag = sys.argv[2]; t0 = time.time(); R = {}; LB = {}
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for s, out, labs in ex.map(_job, [(s, cfgs) for s in ALL]):
            R[s] = out; LB[s] = labs; print(s, f"{time.time()-t0:.0f}s", " ".join(f"{v - out['e7']:+.4f}" for k, v in out.items() if k != "e7"), flush=True)
    lines = []
    for k in R[ALL[0]]:
        means = {w: np.mean([R[s][k] for s in ss]) for w, ss in SESS.items()}; a18 = np.mean([R[s][k] for s in ALL])
        up = sum(R[s][k] > R[s]["e7"] + 1e-9 for s in ALL); dn = sum(R[s][k] < R[s]["e7"] - 1e-9 for s in ALL)
        lines.append(f"{k:85s} " + " ".join(f"{w} {v:.4f}" for w, v in means.items()) + f" | all18 {a18:.4f} | up {up} down {dn}")
    print("\n".join(lines))
    with open(os.path.join(HERE, f"res_{tag}.txt"), "w") as f: f.write("\n".join(lines) + "\n")
    pickle.dump(dict(R=R, labs=LB), open(os.path.join(HERE, f"res_{tag}.pkl"), "wb"))
