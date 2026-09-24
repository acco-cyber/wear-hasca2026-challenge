"""Sim check of final recipe variants on 18 sessions: probs -> mrf4 [-> refine w] [-> block prior beta] -> mrf4.
python sim_final.py <oof.npy> "<w_knn>,<beta>;<w_knn>,<beta>;..." """
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; PLD = r"E:\Claude code\wear\exp\pl"
sys.path.insert(0, TD); sys.path.insert(0, DEC); sys.path.insert(0, PLD)
from pl_labels import SESS, _init, _G

def _job(args):
    s, cfgs = args
    from tlib import session_P, build_X
    from refine_core import knn_label_Q
    from decoder import decode_subject, VARIANTS
    from sim_block import block_prob, apply_block
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; cfg = VARIANTS["mrf4"]
    P = session_P(_G["oof"], st).astype(np.float64); L0 = np.log(np.clip(P, 1e-6, 1))
    def dec(L):
        Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True); return decode_subject(Pn, st, cfg, None)
    lab0 = decode_subject(P, st, cfg, None); X = build_X(d["F"], "v768", 128, 32); out = {}
    for w, beta in cfgs:
        L = L0.copy(); l = lab0
        if w > 0: L = L + w * np.log(knn_label_Q(X, l, 5, 0.9, 0.1)); l = dec(L)
        if beta > 0: L = apply_block(L, block_prob(X, l, "lr", 0.1), beta); l = dec(L)
        out[f"w{w}_b{beta}"] = f1_score(y, l, average="macro")
    return s, out

if __name__ == "__main__":
    oof = sys.argv[1]; cfgs = [tuple(float(x) for x in c.split(",")) for c in sys.argv[2].split(";")]
    allS = sum(SESS.values(), []); R = {}
    with ProcessPoolExecutor(6, initializer=_init, initargs=(oof,)) as ex:
        for s, out in ex.map(_job, [(s, cfgs) for s in allS]): R[s] = out
    for k in R[allS[0]]:
        print(f"{k:12s} " + " ".join(f"{w} {np.mean([R[s][k] for s in ss]):.4f}" for w, ss in SESS.items()) + f" | all18 {np.mean([R[s][k] for s in allS]):.4f}")
