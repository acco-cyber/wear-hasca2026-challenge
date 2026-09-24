"""Session-block prior. WEAR protocol: activities are recorded in two sessions,
block A = {4,5,8,9,10,15,16,17,18}, block B = {1,2,3,6,7,11,12,13,14}. Within a subject, sessions differ in scene/lighting.
Per subject: decode (e7 recipe) -> fit a within-subject block classifier on video features of windows decoded as activities
-> p(block B | window) for every window -> add beta*log p(block) to activity log-probs -> re-decode (mrf4).
python sim_block.py <oof.npy> [grid json]"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import json, time, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; PLD = r"E:\Claude code\wear\exp\pl"
sys.path.insert(0, TD); sys.path.insert(0, DEC); sys.path.insert(0, PLD)
from pl_labels import SESS, _init, _G
NC = 19
BLOCK_B = np.zeros(NC, bool); BLOCK_B[[1, 2, 3, 6, 7, 11, 12, 13, 14]] = True
BLOCK_A = np.zeros(NC, bool); BLOCK_A[[4, 5, 8, 9, 10, 15, 16, 17, 18]] = True

def block_prob(X, lab, method, C=0.1, k=10, alpha=0.9):
    act = lab > 0; yb = BLOCK_B[lab[act]].astype(int)
    if yb.min() == yb.max(): return np.full(len(lab), 0.5)
    if method == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=C, max_iter=2000).fit(X[act], yb); return m.predict_proba(X)[:, 1]
    if method == "knn":
        from refine_core import knn_graph, label_spread
        S_, _ = knn_graph(X, k); Y0 = np.zeros((len(lab), 2)); Y0[np.where(act)[0], yb] = 1
        F = label_spread(S_, Y0, alpha); F = F + 1e-6; return F[:, 1] / F.sum(1)

def apply_block(L, pB, beta, eps=0.02):
    pB = np.clip(pB, eps, 1 - eps); L = L.copy()
    L[:, BLOCK_B] += beta * np.log(pB)[:, None]; L[:, BLOCK_A] += beta * np.log(1 - pB)[:, None]
    return L

def _job(args):
    s, grid = args
    from tlib import session_P, build_X
    from refine_core import knn_label_Q
    from decoder import decode_subject, VARIANTS
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; n = d["n"]; y = d["y"]; cfg = VARIANTS["mrf4"]
    P = session_P(_G["oof"], st).astype(np.float64); L0 = np.log(np.clip(P, 1e-6, 1))
    lab0 = decode_subject(P, st, cfg, None)
    X = build_X(d["F"], "v768", 128, 32)
    Q = knn_label_Q(X, lab0, 5, 0.9, 0.1); L1 = L0 + 6.0 * np.log(Q)
    def dec(L):
        Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True); return decode_subject(Pn, st, cfg, None)
    lab1 = dec(L1); out = {"mrf4": f1_score(y, lab0, average="macro"), "e7": f1_score(y, lab1, average="macro")}
    true_pB = BLOCK_B[y]; act = y > 0
    for g in grid:
        Xb = X if g.get("feat", "v128") == "v128" else build_X(d["F"], "v768", g["feat_d"], 32)
        pB = block_prob(Xb, lab1, g["method"], g.get("C", 0.1), g.get("k", 10), g.get("alpha", 0.9))
        acc = float(((pB > 0.5) == true_pB)[act].mean())
        lab2 = dec(apply_block(L1, pB, g["beta"]))
        key = json.dumps(g); out[key] = f1_score(y, lab2, average="macro"); out["acc|" + key] = acc
    return s, out

if __name__ == "__main__":
    oof = sys.argv[1]
    grid = json.loads(open(sys.argv[2]).read()) if len(sys.argv) > 2 else [
        dict(method="lr", C=0.1, beta=1.0), dict(method="lr", C=0.1, beta=2.0), dict(method="lr", C=0.1, beta=4.0),
        dict(method="lr", C=1.0, beta=2.0), dict(method="knn", k=10, beta=2.0), dict(method="knn", k=30, beta=2.0)]
    allS = sum(SESS.values(), []); t0 = time.time(); R = {}
    with ProcessPoolExecutor(6, initializer=_init, initargs=(oof,)) as ex:
        for s, out in ex.map(_job, [(s, grid) for s in allS]):
            R[s] = out; print(s, {k[:40]: round(v, 4) for k, v in out.items() if not k.startswith("acc|")}, f"{time.time()-t0:.0f}s", flush=True)
    keys = [k for k in R[allS[0]] if not k.startswith("acc|")]
    for k in keys:
        line = " ".join(f"{w} {np.mean([R[s][k] for s in ss]):.4f}" for w, ss in SESS.items())
        acc = np.mean([R[s]["acc|" + k] for s in allS]) if "acc|" + k in R[allS[0]] else float("nan")
        up = sum(R[s][k] > R[s]["e7"] for s in allS)
        print(f"{k:70s} {line} | all18 {np.mean([R[s][k] for s in allS]):.4f} | block acc {acc:.3f} | up vs e7 {up}/18")
