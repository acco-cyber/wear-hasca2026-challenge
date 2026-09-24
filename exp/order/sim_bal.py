"""Protocol-structure count prior on top of e7: per-class count bounds from the decoded family / global counts, re-decode.
python sim_bal.py "<json list of cfgs>" tag
cfg: dict(mode=fam|glob|mix, a, b, rounds=1, knn=0 (re-run kNN refine after), src=L1|L0)"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olib import SESS, ALL, HERE, dec, family_bounds, NC
_G = {}

def _init():
    sys.path.insert(0, HERE)
    from tlib import load_structs
    S = {}
    for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
    _G.update(S=S, C=pickle.load(open(os.path.join(HERE, "cache_e7.pkl"), "rb")))

def run_cfg(c, L0, L1, lab1, st, X):
    from refine_core import knn_label_Q
    lab = lab1; L = L1
    for r in range(c.get("rounds", 1)):
        lo, hi = family_bounds(lab, c["a"], c["b"], mode=c["mode"])
        lab = dec(L, st, lo, hi)
    if c.get("knn", 0):
        L = L0 + 6.0 * np.log(knn_label_Q(X, lab, 5, 0.9, 0.1)); lo, hi = family_bounds(lab, c["a"], c["b"], mode=c["mode"])
        lab = dec(L, st, lo, hi)
    return lab

def _job(args):
    s, cfgs = args
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; L0 = d["L0"].astype(np.float64); L1 = d["L1"].astype(np.float64)
    out = {"e7": f1_score(y, d["lab1"], average="macro")}; labs = {}
    for c in cfgs:
        lab = run_cfg(c, L0, L1, d["lab1"], st, d["X"]); k = json.dumps(c, sort_keys=True)
        out[k] = f1_score(y, lab, average="macro"); labs[k] = lab
    return s, out, labs

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])) if sys.argv[1].endswith(".json") else json.loads(sys.argv[1]); tag = sys.argv[2]; t0 = time.time(); R = {}; LB = {}
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
