"""e7 recipe sim on 18 sessions with alternative link structures.
python sim_xl.py <oof.npy> <struct_tag|orig> [variant list e.g. "e7"] [workers=3]
e7 = mrf4 -> video-kNN label spreading (w6, k5, a.9, eps.1, v768 d128) -> mrf4."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import time, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; XD = r"E:\Claude code\wear\exp\xlimb"
sys.path.insert(0, TD); sys.path.insert(0, DEC); sys.path.insert(0, XD)
SESS = {"eval": ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"],
        "extra": ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"],
        "extra2": ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]}
_G = {}

def _init(oof_path, tag):
    sys.path.insert(0, TD); sys.path.insert(0, DEC); sys.path.insert(0, XD)
    from tlib import load_structs
    C, S = {}, {}
    for w in SESS:
        C.update(pickle.load(open(os.path.join(TD, f"cache_{w}.pkl"), "rb")))
        if tag == "orig": S.update(load_structs(w))
        else: S.update(pickle.load(open(os.path.join(XD, f"struct_{tag}_{w}.pkl"), "rb")))
    for s in S: S[s]["Lm"] = np.asarray(S[s]["Lm"], np.float32)
    o = np.load(oof_path).astype(np.float32); o = o / np.nansum(o, 2, keepdims=True)
    _G.update(C=C, S=S, oof=o)

def run_recipe(P, st, X, variant="e7", extra=None):
    from refine_core import knn_label_Q
    from decoder import decode_subject, VARIANTS
    cfg = VARIANTS["mrf4"]; L0 = np.log(np.clip(P, 1e-6, 1))
    def dec(L):
        Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True); return decode_subject(Pn, st, cfg, None)
    lab0 = decode_subject(P, st, cfg, None)
    if variant == "mrf4": return lab0
    L = L0 + 6.0 * np.log(knn_label_Q(X, lab0, 5, 0.9, 0.1)); return dec(L)

def _job(args):
    s, variants = args
    from tlib import session_P, build_X
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]
    P = session_P(_G["oof"], st).astype(np.float64); X = build_X(d["F"], "v768", 128, 32)
    return s, {v: f1_score(y, run_recipe(P, st, X, v), average="macro") for v in variants}

if __name__ == "__main__":
    oof, tag = sys.argv[1], sys.argv[2]; variants = (sys.argv[3] if len(sys.argv) > 3 else "e7").split(",")
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 3; t0 = time.time()
    allS = sum(SESS.values(), []); R = {}
    with ProcessPoolExecutor(nw, initializer=_init, initargs=(oof, tag)) as ex:
        for s, out in ex.map(_job, [(s, variants) for s in allS]):
            R[s] = out; print(s, {k: round(v, 4) for k, v in out.items()}, f"({time.time()-t0:.0f}s)", flush=True)
    for k in variants:
        print(f"{tag} {k:6s} " + " ".join(f"{w} {np.mean([R[s][k] for s in ss]):.4f}" for w, ss in SESS.items()) + f" | all18 {np.mean([R[s][k] for s in allS]):.4f}")
    pickle.dump(R, open(os.path.join(XD, f"simres_{tag}.pkl"), "wb"))
