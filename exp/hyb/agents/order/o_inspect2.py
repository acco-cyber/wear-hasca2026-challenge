import numpy as np
K = r"E:\Claude code\wear\work\hanbat\keep"
m = np.load(K + r"\sim_meta.npz")
rec, start, y, sbj, fold, sensor = m["rec"], m["start"], m["y"], m["sbj"], m["fold"], m["sensor"]
t = start // 50
l = np.load(K + r"\links_L0.npz")
s = l["oof_succ"]; sc = l["oof_score"]
v = s >= 0
print("succ valid", v.mean(), "range", s[v].min(), s[v].max())
print("same rec", (rec[s[v]] == rec[v]).mean(), "true next", (t[s[v]] == t[v] + 1).mean() if True else 0,
      "same label", (y[s[v]] == y[v]).mean())
print("abs dt median", np.median(np.abs(t[s[v]] - t[v])), "frac |dt|<=5", (np.abs(t[s[v]] - t[v]) <= 5).mean(),
      "<=30", (np.abs(t[s[v]] - t[v]) <= 30).mean(), "<=100", (np.abs(t[s[v]] - t[v]) <= 100).mean())
# in-degree
indeg = np.bincount(s[v], minlength=len(s))
print("indeg dist", np.bincount(indeg)[:6])
e = np.load(K + r"\oof_emb.npy", mmap_mode="r")
print(e.shape, e.dtype, np.linalg.norm(e[:5].astype(np.float32), axis=1))
te = np.load(K + r"\test_emb.npy", mmap_mode="r"); print(te.shape, te.dtype)
b = np.load(K + r"\blend.npz"); ts = b["test_sbj"]; print(np.unique(ts, return_counts=True))
P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy"); print(P.shape, P.dtype, P[:2].sum(1))
PT = np.load(r"E:\Claude code\wear\subs\sub_gl6_p03c03_top2_055_P.npy"); print(PT.shape, PT.dtype, PT[:2].sum(1))
l2 = np.load(K + r"\links_L2_test.npz"); print(l2["succ"][:10], (l2["succ"] >= 0).mean())
