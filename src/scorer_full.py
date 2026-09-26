"""Fit the successor-link scorer on ALL 24 train sessions (2 random-limb draws each) and rebuild the test structure.
Outputs work/scorer_full.pkl and work/test_structure_full.pkl (same format as work/test_structure.pkl)."""
import os, sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chain import make_training_pairs, pair_features, assignment, Scorer
from simulate import load_prep, session_slices

W = r"E:\Claude code\wear\work"; DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep")
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]

def main():
    t0 = time.time(); meta, imu, vid = load_prep(); sl = session_slices(meta); rng = np.random.RandomState(0)
    Fs, Ls = [], []
    for s, (a, b) in sl.items():
        V = np.asarray(vid[a:b], np.float32); I = np.asarray(imu[a:b], np.float32)
        for draw in range(2):
            F, L, cand, limb, Xi = make_training_pairs(V, I, rng)
            F2 = F.reshape(-1, F.shape[-1]); L2 = L.reshape(-1); ok = ~np.isnan(F2[:, 0])
            pos = np.where(ok & (L2 == 1))[0]; neg = np.where(ok & (L2 == 0))[0]
            keep = rng.choice(neg, min(len(neg), 30 * len(pos)), replace=False)
            sel = np.concatenate([pos, keep]); Fs.append(F2[sel]); Ls.append(L2[sel])
        print(f"{s}: pairs ok ({time.time()-t0:.0f}s)", flush=True)
    sc = Scorer(rounds=400).fit(np.concatenate(Fs), np.concatenate(Ls))
    pickle.dump(sc, open(os.path.join(W, "scorer_full.pkl"), "wb")); print("scorer fit", f"{time.time()-t0:.0f}s", flush=True)
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)
    tv = np.load(os.path.join(PREP, "test_vid_pca.npy")); limb = np.array([LIMBS.index(l) for l in tm.sensor_location]); out = {}
    for s in sorted(tm.sbj_id.unique()):
        idx = np.where(tm.sbj_id.to_numpy() == s)[0]; n = len(idx)
        cand, F = pair_features(np.asarray(tv[idx], np.float32), xi[idx], limb[idx])
        lo = sc.logodds(F); succ0, scs, Lm = assignment(cand, lo, n)
        out[s] = dict(idx=idx, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm.astype(np.float16))
        print(f"test sbj {s}: n={n} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(out, open(os.path.join(W, "test_structure_full.pkl"), "wb")); print("saved test_structure_full.pkl")

if __name__ == "__main__":
    main()
