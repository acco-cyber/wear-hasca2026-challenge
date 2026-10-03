import sys, os, time, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
import lightgbm as lgb
from feats import pair_rows, corr_matrices, NPF
HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
z = np.load(os.path.join(HERE, "cache", "sbj_20.npz"), allow_pickle=True)
D = [{k: z[f"{k}{l}"] for k in ("desc", "z_spec", "z_specj", "z_acf", "std", "dyn", "gmean")} for l in range(4)]
bst = lgb.Booster(model_file=os.path.join(HERE, "pair_sec.txt"))
N = len(D[0]["std"]); n = 1_000_000
ia = np.repeat(np.arange(n // N + 1), N)[:n]; ib = np.tile(np.arange(N), n // N + 1)[:n]
t = time.time(); M = corr_matrices(D[0], D[1]); print("corr mats", round(time.time() - t, 2))
t = time.time(); X = pair_rows(D[0], D[1], ia, ib, 0, M=M); print("pair_rows 1M", round(time.time() - t, 2), X.dtype, X.flags["C_CONTIGUOUS"])
t = time.time(); p = bst.predict(X, raw_score=True, num_threads=6); print("predict 1M f32", round(time.time() - t, 2))
X64 = X.astype(np.float64)
t = time.time(); p = bst.predict(X64, raw_score=True, num_threads=6); print("predict 1M f64", round(time.time() - t, 2))
t = time.time(); p = bst.predict(X64, raw_score=True, num_threads=6, num_iteration=150); print("predict 1M f64 150 it", round(time.time() - t, 2))
t = time.time(); p = bst.predict(X64[:200000], raw_score=True, num_threads=6); print("predict 200k f64", round(time.time() - t, 2))
t = time.time(); p = bst.predict(X64[:200000], raw_score=True, num_threads=12); print("predict 200k f64 12thr", round(time.time() - t, 2))
t = time.time(); p = bst.predict(X64[:200000], raw_score=True, num_threads=1); print("predict 200k f64 1thr", round(time.time() - t, 2))
print("lgb version", lgb.__version__, "trees", bst.num_trees())
