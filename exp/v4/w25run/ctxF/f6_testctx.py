"""step 4b: TEST chain-context experts. (i) IMU: the model trained on all training tiles (cache/imu_full.txt) scores every
clean 2025 row; from each of our tiles (anchored by its exact / exact-augmented twin) walk the test chain of its limb.
(ii) ctxB: reached rows that are twins of our tiles of the same limb carry that tile's B2_TEST (per fit).
-> cache/ctx_test.npz (same keys as ctx_oof.npz, pipeline test order; single = rows without chain context),
   cache/ctxB_test_<fit>.npz (W{W}, n{W})"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
from f3b_ctxB import ctxB

T0 = time.time()
cz = np.load(os.path.join(CACHE, "w25_clean.npz")); acc, s25, l25 = cz["acc"], cz["sbj"], cz["limb"]
tz = np.load(os.path.join(CACHE, "test_twins.npz")); twin = tz["twin"]; anch = tz["anchored"]
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); sens = st["sensor_test"].astype(np.int64); tsb = st["test_sbj"].astype(np.int64)
ids = st["ids"]; assert (ids.astype(int) == np.arange(len(ids))).all()          # pipeline test order == data order
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float32)
assert (np.abs(acc[twin] - A26).max() < 1e-6) and (s25[twin] == tsb).all() and (l25[twin] == PIPE_TO_25[sens]).all()
CH = {s: np.load(os.path.join(CACHE, f"test_chain_s{s}.npz")) for s in (22, 23, 24, 25)}
# ---- (ii) ctxB per fit
for fit, d in (("K7", K7), ("K9", K9)):
    sf = np.load(os.path.join(d, "stage.npz"), allow_pickle=True); assert (sf["ids"] == ids).all() and (sf["sensor_test"] == sens).all()
    Bt = sf["B2_TEST"].astype(np.float64); out = {}
    for Wn in (4, 8, 16, 32):
        accB = np.zeros((len(ids), N_CLS)); wsum = np.zeros(len(ids))
        for s in (22, 23, 24, 25):
            z = CH[s]
            for L in range(4):
                rows = z[f"rows{L}"]; n = len(rows); pos_of = np.full(len(acc), -1); pos_of[rows] = np.arange(n)
                mine = np.flatnonzero((tsb == s) & (sens == L)); start = pos_of[twin[mine]]; assert (start >= 0).all()
                mark = np.zeros(n, bool); mark[start] = True; Brows = np.zeros((n, N_CLS)); Brows[start] = Bt[mine]
                a_, w_ = ctxB(Brows, mark, z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), start, Wn)
                accB[mine] = a_; wsum[mine] = w_
        out[f"W{Wn}"] = accB.astype(np.float32); out[f"n{Wn}"] = wsum.astype(np.float32)
        print(f"{fit} test ctxB W={Wn}: mean neighbour weight {wsum.mean():.2f}, share with any {np.mean(wsum > 0):.3f}", flush=True)
    np.savez(os.path.join(CACHE, f"ctxB_test_{fit}.npz"), ids=ids, **out)
# ---- (i) IMU context
if os.path.exists(os.path.join(CACHE, "imu_full.txt")):
    mdl = lgb.Booster(model_file=os.path.join(CACHE, "imu_full.txt"))
    lp25 = np.zeros((len(acc), N_CLS), np.float32)
    for L in range(4):
        r = np.flatnonzero(l25 == PIPE_TO_25[L]); X = tile_feats(acc[r], L)
        lp25[r] = lsm(mdl.predict(X, raw_score=True, num_threads=2).astype(np.float64))
    print(f"IMU scored {len(acc)} rows [{time.time() - T0:.0f}s]", flush=True)
    co = np.load(os.path.join(CACHE, "ctx_oof.npz")); out = {}; single = ~anch
    b7 = st["B2_TEST"].argmax(1); s7 = st["B2_OOF"].argmax(1)
    for key_ in co.files:
        parts = key_.split("_"); Wn = int(parts[0][1:]); dc = float(parts[1][1:]); mode = "prob" if key_.endswith("prob") else "logp"
        c = np.zeros((len(ids), N_CLS), np.float32); wsum = np.zeros(len(ids))
        for s in (22, 23, 24, 25):
            z = CH[s]
            for L in range(4):
                rows = z[f"rows{L}"]; pos_of = np.full(len(acc), -1); pos_of[rows] = np.arange(len(rows))
                mine = np.flatnonzero((tsb == s) & (sens == L)); start = pos_of[twin[mine]]
                cc, ws = context(lp25[rows], z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), start, Wn, decay=dc, mode=mode)
                c[mine] = cc; wsum[mine] = ws
        c[single] = lp25[twin[single]]; out[key_] = c
        print(f"test ctx {key_:14s}: agreement with K7 B2_TEST {np.mean(c.argmax(1) == b7):.4f} (OOF reference {np.mean(co[key_].argmax(1) == s7):.4f}) | mean weight {wsum.mean():.2f}", flush=True)
    np.savez(os.path.join(CACHE, "ctx_test.npz"), ids=ids, single=single, lp25=lp25, **out)
print(f"anchored share {anch.mean():.4f} [{time.time() - T0:.0f}s]")
