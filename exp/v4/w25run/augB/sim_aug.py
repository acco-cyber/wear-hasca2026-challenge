"""Training simulation: does the 2025 augmentation break same-limb chains, and does detect()+restore() repair them?
Per training subject (all sessions pooled, as in the 2025 file) and per ARM limb (legs are never augmented):
  clean     : sim_chain nll3 cost + Hungarian/cycle-break successor (as in research2/w25feas/sim_chain.py)
  aug       : aug.augment(limb rates: left 25 % L; right 20 % R0 + 20 % R1)
  twin25    : only a random 25.5 % of the augmented rows replaced by their clean tile (= what twin replacement gives)
  restored  : aug.detect(limb) + aug.restore on every row (label-free)
Metrics: exact successor accuracy (all links with a true successor) and on 'affected' links (tail or head augmented);
also detect() false positives on the clean tiles.   python sim_aug.py [--subjects 5,9,...] [--jobs 3]"""
import os, sys, time, argparse, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, r"E:\Claude code\wear\exp\v4\research2\w25feas")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_stage, subject_tiles, SENS
from sim_chain import cost_matrix, assign
import aug

D = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="5,9,11,8,6,21,18,7,19,20"); ap.add_argument("--jobs", type=int, default=3)


def run_subject(S, s):
    t0 = time.time(); ii, T = subject_tiles(S, s); n = len(ii)
    pos = np.full(len(S["oof_y"]), -1); pos[ii] = np.arange(n)
    ts = S["true_succ"][ii]; tl = np.where(ts >= 0, pos[np.maximum(ts, 0)], -1); h = tl >= 0
    rng = np.random.default_rng(1000 + s); res = {"sbj": int(s), "n": int(n)}
    for L in (0, 3):                                     # pipeline SENS: 0 right_arm, 3 left_arm
        X = T[L].astype(np.float32)
        Xa, kind = aug.augment(X, rng, limb=SENS[L])
        kd, _ = aug.detect(Xa, limb=SENS[L]); fp, _ = aug.detect(X, limb=SENS[L])
        Xr = aug.restore(Xa, kd)
        a_idx = np.flatnonzero(kind >= 0); sub = a_idx[rng.random(len(a_idx)) < 0.2552]
        Xt = Xa.copy(); Xt[sub] = X[sub]
        aff = h & ((kind >= 0) | (kind[np.maximum(tl, 0)] >= 0))
        r = {"aug_rate": float(np.mean(kind >= 0)), "detect_acc": float(np.mean(kd == kind)), "fp_clean": int(np.sum(fp >= 0)),
             "restore_edge_maxerr": float(np.abs(Xr - X)[:, [0, 1, 2, 47, 48, 49]].max())}
        for nm, Z in (("clean", X), ("aug", Xa), ("twin25", Xt), ("restored", Xr)):
            C = cost_matrix(Z, "nll3"); succ = assign(C)
            tc = C[np.flatnonzero(h), tl[h]]; rank = (C[h] < tc[:, None]).sum(1)
            r[nm] = {"exact": float(np.mean(succ[h] == tl[h])), "exact_affected": float(np.mean(succ[aff] == tl[aff])),
                     "top1_rank": float(np.mean(rank == 0)), "median_rank": float(np.median(rank))}
        res[SENS[L]] = r
    res["sec"] = round(time.time() - t0)
    return res


if __name__ == "__main__":
    a = ap.parse_args()
    S = load_stage(); subs = [int(x) for x in a.subjects.split(",")]
    out = Parallel(n_jobs=a.jobs)(delayed(run_subject)(S, s) for s in subs)
    for r in out:
        for lb in ("right_arm", "left_arm"):
            x = r[lb]
            print(f"sbj {r['sbj']:2d} n {r['n']:5d} {lb:9s} aug {x['aug_rate']:.3f} det_acc {x['detect_acc']:.4f} fp {x['fp_clean']} edge_err {x['restore_edge_maxerr']:.1e} | "
                  + " ".join(f"{nm} {x[nm]['exact']:.3f}/{x[nm]['exact_affected']:.3f}" for nm in ("clean", "aug", "twin25", "restored")) + f" [{r['sec']}s]", flush=True)
    print("\nmean over subjects: exact (all links) / exact (affected links)")
    summ = {}
    for lb in ("right_arm", "left_arm"):
        for nm in ("clean", "aug", "twin25", "restored"):
            e = np.mean([r[lb][nm]["exact"] for r in out]); ea = np.mean([r[lb][nm]["exact_affected"] for r in out])
            summ[f"{lb}_{nm}_exact"] = round(float(e), 4); summ[f"{lb}_{nm}_exact_affected"] = round(float(ea), 4)
            print(f"  {lb:9s} {nm:9s} {e:.4f} / {ea:.4f}")
        summ[f"{lb}_detect_acc"] = round(float(np.mean([r[lb]["detect_acc"] for r in out])), 5)
        summ[f"{lb}_fp_clean_total"] = int(sum(r[lb]["fp_clean"] for r in out))
    print(json.dumps(summ, indent=1))
    json.dump({"per_subject": out, "summary": summ}, open(os.path.join(D, "sim_aug.json"), "w"), indent=1)
