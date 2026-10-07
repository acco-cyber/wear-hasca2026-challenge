"""verify that the pipeline's sensor index = CSV column block order and that OOF rows sit on the CSV 50-sample grid"""
import numpy as np, os
from common import *
S = load_stage()
z = np.load(os.path.join(K7, "tile_scalars.npz")); post = z["oof_post"]; ener = z["oof_ener"]
for s in (5, 9):
    ii, T = subject_tiles(S, s)
    sens = S["sensor_oof"][ii]
    m = T.mean(2)  # (4, n, 3)
    for cand in range(4):
        d = np.abs(m[cand] - post[ii]).mean()
        print(f"sbj {s}: block {cand} mean |tile mean - oof_post| {d:.4f}")
    own = m[sens, np.arange(len(ii))]
    print(f"sbj {s}: own-sensor block vs oof_post: {np.abs(own - post[ii]).mean():.5f}; corr ener vs own std {np.corrcoef(ener[ii], np.linalg.norm(T[sens, np.arange(len(ii))], axis=2).std(1))[0, 1]:.3f}")
    ts = S["true_succ"][ii]; pos = {g: k for k, g in enumerate(ii)}
    loc = np.array([pos.get(g, -1) for g in ts]); print("true succ = next local row:", np.mean(loc[ts >= 0] == np.flatnonzero(ts >= 0) + 1))
