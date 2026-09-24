"""Cache augmented pair features. Fit sessions: R random limb draws (seed 123 stream per session); sim sessions: stored limbs.
python cache_feats.py [R=3]  -> exp/xlimb/feats/<session>_<draw>.npz (cand, F, limb, y)"""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
FD = os.path.join(XD, "feats"); os.makedirs(FD, exist_ok=True)

def draw_limbs(imu4, rng):
    n = len(imu4); limb = rng.randint(0, 4, n)
    bad = np.isnan(imu4[np.arange(n), limb]).any(axis=(1, 2))
    for i in np.where(bad)[0]:
        ok = [l for l in range(4) if not np.isnan(imu4[i, l]).any()]
        if ok: limb[i] = rng.choice(ok)
    return limb

def load_old(w):
    return pickle.load(open(os.path.join(WORK, "sim_struct.pkl") if w == "eval" else os.path.join(TD, f"{w}_struct.pkl"), "rb"))

def _job(args):
    kind, s, R = args; t0 = time.time()
    xl = pickle.load(open(os.path.join(XD, "xl_models.pkl"), "rb")); meta, imu, vid, sl = load_prep(); a, b = sl[s]; n = b - a
    V = np.asarray(vid[a:b], np.float32); W4 = np.asarray(imu[a:b], np.float32); y = meta.y.to_numpy()[a:b]
    if kind == "fit":
        rng = np.random.RandomState(1000 + FIT.index(s)); limbs = [draw_limbs(W4, rng) for _ in range(R)]
    else:
        limbs = [np.asarray(load_old(kind)[s]["limb"])]
    for r, limb in enumerate(limbs):
        p = os.path.join(FD, f"{s}_{r}.npz")
        if os.path.exists(p): continue
        W = W4[np.arange(n), limb]; cand, F = all_pair_features(V, W, limb, xl)
        np.savez(p, cand=cand, F=F.astype(np.float32), limb=limb, y=y)
    return s, time.time() - t0

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if not os.path.exists(os.path.join(XD, "xl_models.pkl")):
        meta, imu, vid, sl = load_prep(); pickle.dump(XLModels().fit(imu, sl, FIT), open(os.path.join(XD, "xl_models.pkl"), "wb"))
    jobs = [("fit", s, R) for s in FIT] + [(w, s, 1) for w in SESS for s in SESS[w]]
    with ProcessPoolExecutor(3) as ex:
        for s, dt in ex.map(_job, jobs): print(s, f"{dt:.0f}s", flush=True)
