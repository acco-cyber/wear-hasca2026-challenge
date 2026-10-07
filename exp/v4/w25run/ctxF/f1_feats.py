"""step 1a: compact IMU features of every training tile position, all four limbs (subject_tiles order) -> cache/imuf_s{s}.npz"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
sys.path.insert(0, FEAS)
from common import load_stage, subject_tiles


def run(S, s):
    t0 = time.time(); p = os.path.join(CACHE, f"imuf_s{s}.npz")
    if os.path.exists(p):
        return f"sbj {s}: cached"
    ii, T = subject_tiles(S, s)
    z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); assert (z["rows"] == ii).all()
    F = np.stack([tile_feats(T[L], L) for L in range(4)])
    np.savez(p, rows=ii, F=F, T=T.astype(np.float32))
    return f"sbj {s}: n={len(ii)} d={F.shape[2]} [{time.time() - t0:.0f}s]"


if __name__ == "__main__":
    S = load_stage(); subs = sorted([int(x) for x in np.unique(S["oof_sbj"])], key=lambda s: -(S["oof_sbj"] == s).sum())
    for m in Parallel(n_jobs=3)(delayed(run)(S, s) for s in subs):
        print(m, flush=True)
