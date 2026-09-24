"""Simulated link structure for EXTRA sessions (not eval, not scorer-fit) with the frozen work/scorer.pkl.
Limbs from a fresh np.random.RandomState(1) per session. -> exp/transductive/extra_struct.pkl
python build_extra.py --sessions sbj_2,sbj_4,sbj_8,sbj_9,sbj_13,sbj_17
"""
import os, sys, time, pickle, argparse
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import make_training_pairs, assignment
from simulate import load_prep, session_slices

WORK = r"E:\Claude code\wear\work"; OUT = r"E:\Claude code\wear\exp\transductive\extra_struct.pkl"

def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", default="sbj_2,sbj_4,sbj_8,sbj_9,sbj_13,sbj_17"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); OUT = a.out
    meta, imu, vid = load_prep(); sl = session_slices(meta)
    sc = pickle.load(open(os.path.join(WORK, "scorer.pkl"), "rb")); sc.m.set_params(n_jobs=3)
    out = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    for s in a.sessions.split(","):
        if s in out: continue
        t0 = time.time(); rng = np.random.RandomState(1)
        aa, bb = sl[s]; n = bb - aa
        F, L, cand, limb, Xi = make_training_pairs(np.asarray(vid[aa:bb], np.float32), np.asarray(imu[aa:bb], np.float32), rng)
        lo = sc.logodds(F); succ0, scs, Lm = assignment(cand, lo, n)
        out[s] = dict(a=aa, b=bb, n=n, y=meta.y.to_numpy()[aa:bb], limb=limb, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm.astype(np.float16))
        print(f"struct {s} n={n} ceiling={L.sum()/(n-1):.3f} ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(out, open(OUT, "wb"))

if __name__ == "__main__":
    main()
