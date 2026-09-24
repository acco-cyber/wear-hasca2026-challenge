"""kNN self-training on top of the decoder agent's mrf4 decoder (exp/decoder/decoder.py, imported read-only).
Per session: P (OOF of the chosen blend, simulated limb) -> mrf4 labels -> [spread labels over within-subject video kNN
-> logP += w*logQ -> mrf4]* . 3 worker processes x 1 thread.
python exp_mrf4.py <oof.npy> <tag>
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
import json, time, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"
EVAL = ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"]
EXTRA = ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"]; EXTRA2 = ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]
DEC_EXTRA = ["sbj_2", "sbj_8", "sbj_13", "sbj_17", "sbj_4", "sbj_11"]          # decoder agent's extra list (lead's 0.8712)
_G = {}

def _init(oof_path):
    sys.path.insert(0, TD); sys.path.insert(0, DEC)
    from tlib import load_structs
    C = {}; S = {}
    for w in ("eval", "extra", "extra2"):
        C.update(pickle.load(open(os.path.join(TD, f"cache_{w}.pkl"), "rb"))); S.update(load_structs(w))
    o = np.load(oof_path).astype(np.float32); o = o / np.nansum(o, 2, keepdims=True)
    _G.update(C=C, S=S, oof=o)

def _job(args):
    s, cfg = args
    from tlib import build_X, session_P, NC
    from refine_core import knn_graph, label_spread
    from decoder import decode_subject, VARIANTS
    from sklearn.metrics import f1_score
    d = _G["C"][s]; st = _G["S"][s]; y = d["y"]; n = d["n"]
    P = session_P(_G["oof"], st).astype(np.float64); dcfg = VARIANTS[cfg.get("variant", "mrf4")]
    lab = decode_subject(P, st, dcfg, None); out = [f1_score(y, lab, average="macro")]; qacc = []
    L = np.log(np.clip(P, 1e-6, 1)); Xc = {}
    for step in cfg["steps"]:
        key = (step.get("d_vid", 64),)
        if key not in Xc: Xc[key] = build_X(d["F"], "v768", step.get("d_vid", 64), 32)
        S_, _ = knn_graph(Xc[key], step.get("k", 5)); Y0 = np.zeros((n, NC)); Y0[np.arange(n), lab] = 1
        F = label_spread(S_, Y0, step.get("alpha", 0.9)); Q = F / np.maximum(F.sum(1, keepdims=True), 1e-12)
        qacc.append(float((Q.argmax(1) == y).mean()))
        eps = step.get("eps", 0.1); Q = (1 - eps) * Q + eps / NC
        L = L + step["w"] * np.log(Q); Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        lab = decode_subject(Pn, st, dcfg, None); out.append(f1_score(y, lab, average="macro"))
    return s, out, qacc

def main():
    oof_path, tag = sys.argv[1], sys.argv[2]
    K = lambda w, d=64, k=5, al=0.9: dict(w=w, d_vid=d, k=k, alpha=al)
    G = [dict(steps=[K(2.0)]), dict(steps=[K(6.0)]), dict(steps=[K(6.0, 128)]), dict(steps=[K(10.0)]),
         dict(steps=[K(6.0), K(6.0)]), dict(steps=[K(2.0), K(2.0)])]
    if len(sys.argv) > 3: G = json.loads(open(sys.argv[3]).read())
    sessions = EVAL + EXTRA + EXTRA2; rows = []; t0 = time.time()
    with ProcessPoolExecutor(3, initializer=_init, initargs=(oof_path,)) as ex:
        for ci, cfg in enumerate(G):
            res = {s: (f, q) for s, f, q in ex.map(_job, [(s, cfg) for s in sessions])}
            nr = len(res[sessions[0]][0])
            print(f"[{ci+1}/{len(G)} {time.time()-t0:.0f}s] {json.dumps(cfg)}", flush=True)
            for r in range(nr):
                row = dict(cfg=json.dumps(cfg), round=r, **{s: round(res[s][0][r], 4) for s in sessions})
                for nm, ss in (("eval", EVAL), ("extra", EXTRA), ("extra2", EXTRA2), ("dec_extra", DEC_EXTRA), ("all18", sessions)):
                    row[nm] = round(np.mean([res[s][0][r] for s in ss]), 4); row["d_" + nm] = round(row[nm] - np.mean([res[s][0][0] for s in ss]), 4)
                row["n_up"] = int(sum(res[s][0][r] > res[s][0][0] for s in sessions)); row["n_down"] = int(sum(res[s][0][r] < res[s][0][0] for s in sessions))
                row["q_acc"] = round(np.mean([res[s][1][r - 1] for s in sessions]), 3) if r > 0 else np.nan
                rows.append(row)
                print(f"   r{r}: eval {row['eval']:.4f} ({row['d_eval']:+.4f}) dec_extra {row['dec_extra']:.4f} ({row['d_dec_extra']:+.4f}) "
                      f"extra {row['extra']:.4f} ({row['d_extra']:+.4f}) extra2 {row['extra2']:.4f} ({row['d_extra2']:+.4f}) up/down {row['n_up']}/{row['n_down']} q_acc {row['q_acc']}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(TD, f"results_mrf4_{tag}.csv"), index=False)

if __name__ == "__main__":
    main()
