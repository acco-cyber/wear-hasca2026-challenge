import os, numpy as np
C = r"E:\Claude code\wear\exp\v4\w25run\testD\cache"
z = np.load(os.path.join(C, "chaindeaug_s5.npz"))
for k in z.files: print("sim", k, z[k].shape, z[k].dtype, z[k].ravel()[:8])
t = np.load(os.path.join(C, "testdeaug_chain_s24.npz"))
for k in t.files: print("test", k, t[k].shape, t[k].dtype, t[k].ravel()[:8])
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
for k in st.files: print("stage", k, st[k].shape, st[k].dtype)
L = np.load(r"E:\Claude code\wear\exp\v4\w25run\testD\links_oof_K7_deaug.npz")
for k in L.files: print("links", k, L[k].shape, L[k].dtype)
lz = np.load(os.path.join(K7, "link_logodds.npz"))
print([k for k in lz.files][:12], len(lz.files))
print(lz["oof_5_cand"].shape, lz["oof_5_L"].shape, lz["oof_5_cand"][:2,:5], lz["oof_5_L"][:2,:5])
lk = np.load(os.path.join(K7, "links.npz"))
for k in lk.files: print("k7links", k, lk[k].shape)
print(np.percentile(lk["qn_ref"], [1, 50, 90, 99, 99.9]))
