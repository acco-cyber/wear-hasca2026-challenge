"""Sim harness: run refinement configs on the 6 eval + 6 extra sessions (3 worker processes x 1 thread).
python exp_refine.py <grid_name>   (grids defined in GRIDS below) -> results_<grid>.csv (one row per config x round)
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, time, pickle, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

TD = r"E:\Claude code\wear\exp\transductive"
_C = None

def _init():
    global _C
    sys.path.insert(0, TD)
    _C = pickle.load(open(os.path.join(TD, "cache_eval.pkl"), "rb")); _C.update(pickle.load(open(os.path.join(TD, "cache_extra.pkl"), "rb")))
    if os.path.exists(os.path.join(TD, "cache_extra2.pkl")): _C.update(pickle.load(open(os.path.join(TD, "cache_extra2.pkl"), "rb")))

def _job(args):
    s, cfg = args
    from refine_core import refine
    d = _C[s]; log = []
    labs, _ = refine(d["P"], d["F"], d["g"], d["chains"], d["lab0"], d["Pg0"], cfg, y=d["y"], log=log)
    return s, [f1_score(d["y"], l, average="macro") for l in labs], log

EVAL = ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"]; EXTRA = ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"]
EXTRA2 = ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"] if os.path.exists(os.path.join(TD, "cache_extra2.pkl")) else []

def grid(name):
    G = []
    if name == "g1":
        for spec in ("v768", "vm+vs+vd+imu", "v768+imu"):
            for w in (0.3, 0.6, 1.0):
                G.append(dict(clf="lr", spec=spec, w=w, sel="marg", q=0.5, rounds=2))
        for w in (0.3, 0.6, 1.0):
            G.append(dict(clf="knn", spec="v768", k=20, alpha=0.9, w=w, sel="marg", q=0.5, rounds=2))
            G.append(dict(clf="ksmooth", spec="v768", k=20, alpha=0.8, w=w, rounds=1))
            G.append(dict(clf="lda", spec="v768", w=w, sel="marg", q=0.5, rounds=2))
    elif name.startswith("json:"):
        G = json.load(open(name[5:]))
    return G

def main():
    name = sys.argv[1]; G = grid(name); sessions = EVAL + EXTRA + EXTRA2
    out_csv = os.path.join(TD, f"results_{name.replace('json:', '').replace(os.sep, '_').replace(':', '')[-40:]}.csv")
    rows = []; t0 = time.time()
    with ProcessPoolExecutor(3, initializer=_init) as ex:
        for ci, cfg in enumerate(G):
            res = list(ex.map(_job, [(s, cfg) for s in sessions]))
            f = {s: fs for s, fs, _ in res}; logs = {s: lg for s, _, lg in res}
            nr = len(next(iter(f.values())))
            for r in range(nr):
                row = dict(cfg=json.dumps(cfg), round=r)
                for s in sessions: row[s] = round(f[s][r], 4)
                row["eval"] = round(np.mean([f[s][r] for s in EVAL]), 4); row["extra"] = round(np.mean([f[s][r] for s in EXTRA]), 4)
                row["d_eval"] = round(row["eval"] - np.mean([f[s][0] for s in EVAL]), 4); row["d_extra"] = round(row["extra"] - np.mean([f[s][0] for s in EXTRA]), 4)
                if EXTRA2:
                    row["extra2"] = round(np.mean([f[s][r] for s in EXTRA2]), 4); row["d_extra2"] = round(row["extra2"] - np.mean([f[s][0] for s in EXTRA2]), 4)
                row["n_up"] = int(sum(f[s][r] > f[s][0] for s in sessions)); row["n_down"] = int(sum(f[s][r] < f[s][0] for s in sessions))
                if r > 0:
                    row["sel_acc"] = round(np.mean([logs[s][r - 1]["sel_acc"] for s in sessions]), 3)
                    row["q_acc"] = round(np.mean([logs[s][r - 1]["q_acc"] for s in sessions]), 3)
                rows.append(row)
            last = [x for x in rows if x["cfg"] == json.dumps(cfg)]
            print(f"[{ci+1}/{len(G)} {time.time()-t0:.0f}s] {json.dumps(cfg)}", flush=True)
            for x in last: print(f"   r{x['round']}: eval {x['eval']:.4f} ({x['d_eval']:+.4f})  extra {x['extra']:.4f} ({x['d_extra']:+.4f})  x2 {x.get('extra2', 0):.4f} ({x.get('d_extra2', 0):+.4f})  up/down {x['n_up']}/{x['n_down']}  q_acc {x.get('q_acc', '')}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
