"""Hierarchical (family-then-variant) decode on top of the e7 log-scores L1.
Stage 1: mrf4 on family super-classes (family log-prob = logsumexp over its variants, placed on a representative class,
         bounds K*[lo,hi]); Stage 2: mrf4 on L1 restricted to the stage-1 family of each window.
python sim_hier.py cfgs.json tag"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olib import SESS, ALL, HERE, dec, NC, FAMS
_G = {}
FAMLIST = [[0]] + [v for v in FAMS.values()] + [[15], [18]]
FID = np.zeros(NC, int)
for i, cls in enumerate(FAMLIST): FID[cls] = i
NEG = -30.0

def _init():
    sys.path.insert(0, HERE)
    from tlib import load_structs
    S = {}
    for w in ("eval", "extra", "extra2"): S.update(load_structs(w))
    _G.update(S=S, C=pickle.load(open(os.path.join(HERE, "cache_e7.pkl"), "rb")))

def hier(L, st, c, lab_init=None):
    lo0, hi0 = 80.0, 250.0
    Lf = np.full_like(L, NEG); lo = np.zeros(NC - 1); hi = np.full(NC - 1, hi0)
    for cls in FAMLIST:
        r = cls[0]; Lf[:, r] = np.logaddexp.reduce(L[:, cls], 1) + c.get("fam_bonus", 0.0) * (r > 0) * (len(cls) > 1)
        if r > 0: lo[r - 1] = c.get("flo", 1.0) * lo0 * len(cls); hi[r - 1] = hi0 * len(cls)
    labf = dec(Lf, st, lo, hi); fam = FID[labf]
    if c.get("stage2", "restrict") == "restrict":
        L2 = L.copy(); L2[FID[None, :] != fam[:, None]] = NEG
        return dec(L2, st)
    if c["stage2"] == "argmax":
        L2 = L.copy(); L2[FID[None, :] != fam[:, None]] = NEG; return L2.argmax(1)
    if c["stage2"] == "soft":         # soft family prior: penalise classes outside the stage-1 family by delta
        L2 = L.copy(); L2[FID[None, :] != fam[:, None]] -= c["delta"]; return dec(L2, st)

def _job(args):
    s, cfgs = args
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; L1 = d["L1"].astype(np.float64)
    out = {"e7": f1_score(y, d["lab1"], average="macro")}; labs = {}
    for c in cfgs:
        lab = hier(L1, st, c); k = json.dumps(c, sort_keys=True)
        out[k] = f1_score(y, lab, average="macro"); labs[k] = lab
    return s, out, labs

if __name__ == "__main__":
    cfgs = json.load(open(sys.argv[1])); tag = sys.argv[2]; t0 = time.time(); R = {}; LB = {}
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for s, out, labs in ex.map(_job, [(s, cfgs) for s in ALL]):
            R[s] = out; LB[s] = labs; print(s, f"{time.time()-t0:.0f}s", {k[:30]: round(v, 4) for k, v in out.items()}, flush=True)
    lines = []
    for k in R[ALL[0]]:
        means = {w: np.mean([R[s][k] for s in ss]) for w, ss in SESS.items()}; a18 = np.mean([R[s][k] for s in ALL])
        up = sum(R[s][k] > R[s]["e7"] + 1e-9 for s in ALL); dn = sum(R[s][k] < R[s]["e7"] - 1e-9 for s in ALL)
        lines.append(f"{k:75s} " + " ".join(f"{w} {v:.4f}" for w, v in means.items()) + f" | all18 {a18:.4f} | up {up} down {dn}")
    print("\n".join(lines))
    with open(os.path.join(HERE, f"res_{tag}.txt"), "w") as f: f.write("\n".join(lines) + "\n")
    pickle.dump(dict(R=R, labs=LB), open(os.path.join(HERE, f"res_{tag}.pkl"), "wb"))
