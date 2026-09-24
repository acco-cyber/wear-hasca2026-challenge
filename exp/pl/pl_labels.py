"""Step 1 of pseudo-label self-training: decoded labels (e7 recipe = mrf4 + within-subject video-kNN refinement, w6 d128)
for the 18 sim sessions from a given OOF file. Saves exp/pl/labels_<tag>.pkl: session -> dict(a, n, limb, lab, y).
python pl_labels.py <oof.npy> <tag>"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import pickle, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; OUT = r"E:\Claude code\wear\exp\pl"
sys.path.insert(0, TD); sys.path.insert(0, DEC)
SESS = {"eval": ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"],
        "extra": ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"],
        "extra2": ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]}
_G = {}

def _init(oof_path):
    sys.path.insert(0, TD); sys.path.insert(0, DEC)
    from tlib import load_structs
    C, S = {}, {}
    for w in SESS:
        C.update(pickle.load(open(os.path.join(TD, f"cache_{w}.pkl"), "rb"))); S.update(load_structs(w))
    o = np.load(oof_path).astype(np.float32); o = o / np.nansum(o, 2, keepdims=True)
    _G.update(C=C, S=S, oof=o)

def decode_refined(P, F, st, n, w=6.0, d_vid=128, k=5, alpha=0.9, eps=0.1, variant="mrf4"):
    from tlib import build_X, NC
    from refine_core import knn_graph, label_spread
    from decoder import decode_subject, VARIANTS
    dcfg = VARIANTS[variant]
    lab0 = decode_subject(P, st, dcfg, None)
    X = build_X(F, "v768", d_vid, 32); S_, _ = knn_graph(X, k)
    Y0 = np.zeros((n, NC)); Y0[np.arange(n), lab0] = 1
    Fq = label_spread(S_, Y0, alpha); Q = Fq / np.maximum(Fq.sum(1, keepdims=True), 1e-12); Q = (1 - eps) * Q + eps / NC
    L = np.log(np.clip(P, 1e-6, 1)) + w * np.log(Q); Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
    return decode_subject(Pn, st, dcfg, None), lab0

def _job(s):
    from tlib import session_P
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; n = d["n"]
    P = session_P(_G["oof"], st).astype(np.float64)
    lab, lab0 = decode_refined(P, d["F"], st, n)
    y = d["y"]
    return s, dict(a=int(st["a"]), n=int(n), limb=np.asarray(st["limb"]), lab=lab, y=y,
                   f1=float(f1_score(y, lab, average="macro")), f1_mrf4=float(f1_score(y, lab0, average="macro")))

if __name__ == "__main__":
    oof_path, tag = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    allS = sum(SESS.values(), []); res = {}
    with ProcessPoolExecutor(6, initializer=_init, initargs=(oof_path,)) as ex:
        for s, r in ex.map(_job, allS):
            res[s] = r; print(f"{s}: refined F1 {r['f1']:.4f} (mrf4 {r['f1_mrf4']:.4f}) label acc {np.mean(r['lab']==r['y']):.3f} ({time.time()-t0:.0f}s)", flush=True)
    for w, ss in SESS.items():
        print(w, "mean refined", round(np.mean([res[s]["f1"] for s in ss]), 4), "mrf4", round(np.mean([res[s]["f1_mrf4"] for s in ss]), 4))
    pickle.dump(res, open(os.path.join(OUT, f"labels_{tag}.pkl"), "wb")); print("saved")
