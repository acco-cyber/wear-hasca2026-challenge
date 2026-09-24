"""Structured transition prior (chain Viterbi) / structured ICM coupling in the final mrf4 decode of the e7 recipe.
Transition prior fit on the scorer-fit sessions only (sbj_1,3,7,12,16,19; true 1-s label sequences of train_meta).
python sim_T.py cfgs.json tag     cfg: dict(T=uniform|emp|null, eps, r, kappa, both=0/1 (also use in the first decode))"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olib import SESS, ALL, HERE, NC, FAMS, to_P
_G = {}
FIT = ["sbj_1", "sbj_3", "sbj_7", "sbj_12", "sbj_16", "sbj_19"]

def fit_T():
    import pandas as pd
    meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv"); T = np.zeros((NC, NC))
    for s in FIT:
        y = meta[meta.session == s].y.to_numpy(); np.add.at(T, (y[:-1], y[1:]), 1)
    return T

def make_logT(c, Temp, ps=0.7):
    if c["T"] == "uniform": return None
    off = Temp.copy(); np.fill_diagonal(off, 0)
    if c["T"] == "emp":
        off = off / np.maximum(off.sum(1, keepdims=True), 1)
        off = (1 - c["eps"]) * off + c["eps"] / (NC - 1); np.fill_diagonal(off, 0); off = off / off.sum(1, keepdims=True)
    elif c["T"] == "null":          # act<->null switches r times as likely as act<->act switches
        off = np.ones((NC, NC)); off[1:, 0] = c["r"]; off[0, 1:] = 1.0; np.fill_diagonal(off, 0); off = off / off.sum(1, keepdims=True)
    T = (1 - ps) * off; np.fill_diagonal(T, ps); return np.log(np.maximum(T, 1e-9))

def make_M(c):
    k = c.get("kappa", 0.0)
    if k == 0: return None
    M = np.eye(NC)
    for cls in FAMS.values():
        for a in cls:
            for b in cls:
                if a != b: M[a, b] = k
    return M

def _init():
    sys.path.insert(0, HERE)
    from tlib import load_structs
    S = {}
    for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
    _G.update(S=S, C=pickle.load(open(os.path.join(HERE, "cache_e7.pkl"), "rb")), T=fit_T())

def _job(args):
    s, cfgs = args
    from sklearn.metrics import f1_score
    from mydec import decode, prep
    from refine_core import knn_label_Q
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; L0 = d["L0"].astype(np.float64); L1 = d["L1"].astype(np.float64)
    pre = prep(st, len(y))
    out = {"e7": f1_score(y, d["lab1"], average="macro")}; labs = {}
    for c in cfgs:
        logT = make_logT(c, _G["T"]); M = make_M(c)
        if c.get("both", 0):
            l0 = decode(to_P(L0), st, pre, logT, M); L = L0 + 6.0 * np.log(knn_label_Q(d["X"], l0, 5, 0.9, 0.1))
        else: L = L1
        lab = decode(to_P(L), st, pre, logT, M); k = json.dumps(c, sort_keys=True)
        out[k] = f1_score(y, lab, average="macro"); labs[k] = lab
    return s, out, labs

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); tag = sys.argv[2]; t0 = time.time(); R = {}; LB = {}
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for s, out, labs in ex.map(_job, [(s, cfgs) for s in ALL]):
            R[s] = out; LB[s] = labs; print(s, f"{time.time()-t0:.0f}s", flush=True)
    lines = []
    for k in R[ALL[0]]:
        means = {w: np.mean([R[s][k] for s in ss]) for w, ss in SESS.items()}; a18 = np.mean([R[s][k] for s in ALL])
        up = sum(R[s][k] > R[s]["e7"] + 1e-9 for s in ALL); dn = sum(R[s][k] < R[s]["e7"] - 1e-9 for s in ALL)
        lines.append(f"{k:75s} " + " ".join(f"{w} {v:.4f}" for w, v in means.items()) + f" | all18 {a18:.4f} | up {up} down {dn}")
    print("\n".join(lines))
    with open(os.path.join(HERE, f"res_{tag}.txt"), "w") as f: f.write("\n".join(lines) + "\n")
    pickle.dump(dict(R=R, labs=LB), open(os.path.join(HERE, f"res_{tag}.pkl"), "wb"))
