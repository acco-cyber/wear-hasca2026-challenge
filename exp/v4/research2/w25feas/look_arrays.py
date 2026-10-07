import os, numpy as np
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
for k in st.files:
    v = st[k]; print("stage", k, v.shape, v.dtype, (v[:5] if v.ndim == 1 else ""))
lk = np.load(os.path.join(K7, "links.npz"))
for k in lk.files:
    print("links", k, lk[k].shape, lk[k].dtype)
z = np.load(os.path.join(K7, "link_logodds.npz"))
print("logodds keys", z.files[:12], len(z.files))
for k in z.files[:6]:
    print(k, z[k].shape, z[k].dtype)
ts = np.load(os.path.join(K7, "tile_scalars.npz"))
for k in ts.files:
    print("scal", k, ts[k].shape, ts[k].dtype)
d = np.load(os.path.join(K7, "dec_cache.npz"), allow_pickle=True)
for k in d.files:
    print("dec", k, d[k].shape, d[k].dtype)
y = st["oof_y"]; sb = st["oof_sbj"]; rec = st["oof_rec"]; s0 = st["oof_start"]; tsu = st["true_succ"]
print("subjects", np.unique(sb), "recs", np.unique(rec))
for r in np.unique(rec)[:3]:
    ii = np.flatnonzero(rec == r); print(r, len(ii), s0[ii].min(), s0[ii].max(), np.unique(np.diff(np.sort(s0[ii])))[:10], np.unique(sb[ii]))
print("true_succ linked", (tsu >= 0).mean())
print("sensor_oof", np.bincount(st["sensor_oof"].astype(int)))
