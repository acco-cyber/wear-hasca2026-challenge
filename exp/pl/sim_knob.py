"""Sim of e7 recipe with mrf4 config overrides.  python sim_knob.py <oof.npy> '<json list of override dicts>' [w_knn]"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; PLD = r"E:\Claude code\wear\exp\pl"
sys.path.insert(0, TD); sys.path.insert(0, DEC); sys.path.insert(0, PLD)
from pl_labels import SESS, _init, _G

def _job(args):
    s, ovs, w = args
    from tlib import session_P, build_X
    from refine_core import knn_label_Q
    from decoder import decode_subject, VARIANTS
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]
    P = session_P(_G["oof"], st).astype(np.float64); L0 = np.log(np.clip(P, 1e-6, 1)); X = build_X(d["F"], "v768", 128, 32); out = {}
    for ov in ovs:
        cfg = dict(VARIANTS["mrf4"], **ov)
        l = decode_subject(P, st, cfg, None)
        if w > 0:
            L = L0 + w * np.log(knn_label_Q(X, l, 5, 0.9, 0.1)); Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
            l = decode_subject(Pn, st, cfg, None)
        out[json.dumps(ov)] = f1_score(y, l, average="macro")
    return s, out

if __name__ == "__main__":
    oof = sys.argv[1]; ovs = json.loads(sys.argv[2]); w = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    allS = sum(SESS.values(), []); R = {}
    with ProcessPoolExecutor(6, initializer=_init, initargs=(oof,)) as ex:
        for s, out in ex.map(_job, [(s, ovs, w) for s in allS]): R[s] = out
    for k in R[allS[0]]:
        print(f"{k:40s} " + " ".join(f"{g} {np.mean([R[s][k] for s in ss]):.4f}" for g, ss in SESS.items()) + f" | all18 {np.mean([R[s][k] for s in allS]):.4f}", flush=True)
