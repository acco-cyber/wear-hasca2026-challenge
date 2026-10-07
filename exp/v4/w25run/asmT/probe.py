import numpy as np, os
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"; K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
TD = r"E:\Claude code\wear\exp\v4\w25run\testD"
a = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); b = np.load(os.path.join(K9, "stage.npz"), allow_pickle=True)
for k in ["oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "true_succ", "sensor_oof", "test_sbj", "sensor_test"]:
    print(k, k in b.files, (a[k] == b[k]).all() if k in b.files else None, a[k].shape)
for K in (K7, K9):
    r = np.load(os.path.join(K, "links.npz")); print(r.files, np.percentile(r["qn_ref"], [1, 50, 90, 99]))
    z = np.load(os.path.join(K, "link_logodds.npz")); fs = z.files; print(len(fs), fs[:6], [f for f in fs if f.startswith("test_")][:8])
for f in ["links_test_K7_deaug.npz", "links_test_K9_deaug.npz", "links_oof_K9_deaug.npz", "links_oof_K7_deaug.npz"]:
    z = np.load(os.path.join(TD, f)); print(f, {k: z[k].shape for k in z.files})
    if "test_score" in z.files:
        s = z["test_score"]; print(" test score pct", np.percentile(s[s > -49], [1, 50, 90, 99]))
    s = z["oof_score"]; print(" oof score pct", np.percentile(s[s > -49], [1, 50, 90, 99]))
for F in ("K7", "K9"):
    t = np.load(os.path.join(TD, f"links_test_{F}_deaug.npz")); o = np.load(os.path.join(TD, f"links_oof_{F}_deaug.npz"))
    print(F, "oof in test file == links_oof:", (t["oof_succ"] == o["oof_succ"]).all(), np.abs(t["oof_score"] - o["oof_score"]).max())
