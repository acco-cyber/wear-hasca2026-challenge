"""check rlib.rows == v4_local.refiner_rows (H=3) on one matching, and rlib.flips == refiner_flips"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
import rlib as R

d = R.load_cache()
lP = np.log(np.clip(d["Po"].astype(np.float64), 1e-9, None)).astype(np.float32)
lQ = np.log(d["Qo"].astype(np.float64) + 1e-9).astype(np.float32)
su = d["Lo_s"][3]
t = time.time()
X, ix = R.rows(d["fin_o"], su, d["Bo"], lP, lQ, d["lwo"], d["sens_o"], d["ener_o"], d["post_o"], d["vmot_o"], d["vmean_o"])
print("vectorised", X.shape, f"{time.time() - t:.1f}s")
t = time.time()
Xr, tr, orr = V.refiner_rows(d["fin_o"], su, d["Bo"], d["Po"], d["Qo"], d["lwo"], d["sens_o"], d["ener_o"], d["post_o"], d["vmot_o"], d["vmean_o"])
print("reference", Xr.shape, f"{time.time() - t:.1f}s")
print("tiles equal", np.array_equal(tr, ix["g"]), "others equal", np.array_equal(orr, ix["oth"]))
diff = np.abs(X - Xr).max(0); print("max abs diff per feature", np.round(diff, 5))
p = np.random.RandomState(0).rand(len(tr))
a, na = V.refiner_flips(d["fin_o"], tr, orr, p, 1, 0.5); b, nb = R.flips(d["fin_o"], tr, orr, p, 1, 0.5)
print("flips equal", np.array_equal(a, b), na, nb)
