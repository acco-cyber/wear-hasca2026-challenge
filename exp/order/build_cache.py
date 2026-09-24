"""Cache the e7 stage (L1 log-scores, lab0, lab1, X) for the 18 sim sessions -> exp/order/cache_e7.pkl."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olib import SESS, ALL, OOF, HERE, e7_stage
from pl_labels import _init, _G

def _job(s):
    from tlib import session_P
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; P = session_P(_G["oof"], st).astype(np.float64)
    t0 = time.time(); L0, L1, lab0, lab1, X = e7_stage(P, d["F"], st)
    return s, dict(L0=L0.astype(np.float32), L1=L1.astype(np.float32), lab0=lab0, lab1=lab1, X=X.astype(np.float32), y=d["y"],
                   f1=f1_score(d["y"], lab1, average="macro"), sec=time.time() - t0)

if __name__ == "__main__":
    R = {}
    with ProcessPoolExecutor(4, initializer=_init, initargs=(OOF,)) as ex:
        for s, r in ex.map(_job, ALL): R[s] = r; print(s, round(r["f1"], 4), f"{r['sec']:.0f}s", flush=True)
    for w, ss in SESS.items(): print(w, round(np.mean([R[s]["f1"] for s in ss]), 4))
    print("all18", round(np.mean([R[s]["f1"] for s in ALL]), 4))
    pickle.dump(R, open(os.path.join(HERE, "cache_e7.pkl"), "wb"))
