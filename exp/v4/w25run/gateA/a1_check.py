"""Gate A: sanity checks before building the decode link files (K7 vs K9 stage arrays, qn_ref, cached sim links)."""
import os, numpy as np
W = r"E:\Claude code\wear"
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
C = os.path.join(W, "exp", "v4", "research2", "w25feas", "cache")
s7 = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); s9 = np.load(os.path.join(K9, "stage.npz"), allow_pickle=True)
for k in ["oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "true_succ", "sensor_oof", "test_sbj", "sensor_test", "ids"]:
    a, b = s7[k], s9[k]
    print(k, a.shape, a.dtype, "equal" if (a.shape == b.shape and (a == b).all()) else "DIFF")
l7 = np.load(os.path.join(K7, "links.npz")); l9 = np.load(os.path.join(K9, "links.npz"))
print("links keys", l7.files)
for k in l7.files:
    print(k, l7[k].shape, l7[k].dtype)
print("qn_ref equal:", l7["qn_ref"].shape == l9["qn_ref"].shape and np.allclose(l7["qn_ref"], l9["qn_ref"]),
      "K7 q", np.quantile(l7["qn_ref"], [0.1, 0.5, 0.9]).round(3), "K9 q", np.quantile(l9["qn_ref"], [0.1, 0.5, 0.9]).round(3))
ts = s7["true_succ"].astype(np.int64); h = ts >= 0
for nm, l in (("K7", l7), ("K9", l9)):
    print(nm, "own exact (over ts>=0):", np.round([(l["oof_succ"][k][h] == ts[h]).mean() for k in range(len(l["oof_succ"]))], 4))
z = np.load(os.path.join(C, "links_lgb_a1.npz"))
print("sim links keys", z.files, z["oof_succ"].shape, z["oof_score"].shape, z["oof_score"].dtype)
print("sim exact (over ts>=0):", np.round([(z["oof_succ"][k][h] == ts[h]).mean() for k in range(len(z["oof_succ"]))], 4))
m = z["oof_succ"][0] >= 0
print("sim linked share", m.mean().round(4), "own linked share", (l7["oof_succ"][0] >= 0).mean().round(4))
print("score quantiles sim", np.quantile(z["oof_score"][0][m], [0.1, 0.5, 0.9]).round(3), "own", np.quantile(l7["oof_score"][0][l7["oof_succ"][0] >= 0], [0.1, 0.5, 0.9]).round(3))
