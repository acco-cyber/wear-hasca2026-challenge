import os, sys, pickle, time
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, pandas as pd
t0 = time.time()
meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
print(meta.head()); print(meta.session.unique())
sl = {s: (g.index.min(), g.index.max() + 1) for s, g in meta.groupby("session", sort=False)}
print(sl)
imu = np.load(r"E:\Claude code\wear\data\prep\train_imu.npy", mmap_mode="r")
print(imu.shape, imu.dtype, np.nanmean(np.abs(np.asarray(imu[:2000], np.float32))))
S = pickle.load(open(r"E:\Claude code\wear\work\sim_struct.pkl", "rb"))
for s, st in S.items(): print(s, {k: (getattr(v, "shape", v) if not np.isscalar(v) else v) for k, v in st.items()})
for w in ("extra", "extra2"):
    S2 = pickle.load(open(rf"E:\Claude code\wear\exp\transductive\{w}_struct.pkl", "rb"))
    for s, st in S2.items(): print(w, s, st["a"], st["b"], st["cand"].shape, st["lo"].dtype)
T = pickle.load(open(r"E:\Claude code\wear\work\test_structure.pkl", "rb"))
for s, st in T.items(): print("test", s, {k: getattr(v, "shape", v) for k, v in st.items()}, st["lo"].dtype, st["Lm"].dtype)
C = pickle.load(open(r"E:\Claude code\wear\exp\transductive\cache_eval.pkl", "rb"))
for s, d in C.items(): print("cache", s, {k: (getattr(v, "shape", v) if not isinstance(v, dict) else list(v.keys())) for k, v in d.items()}); break
print(time.time() - t0)
