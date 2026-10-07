import numpy as np, os
C = r"E:\Claude code\wear\exp\v4\w25run\testD\cache"
z = np.load(os.path.join(C, "chaindeaug_s0.npz"))
for k in z.files:
    print(k, z[k].shape, z[k].dtype, z[k][:5] if z[k].ndim == 1 else z[k][:, :5])
t = np.load(os.path.join(C, "testdeaug_chain_s22.npz"))
for k in t.files:
    print("T", k, t[k].shape, t[k].dtype, t[k][:5] if t[k].ndim == 1 else t[k][:, :5])
S = np.load(r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4\stage.npz", allow_pickle=True)
for k in S.files:
    print("S", k, S[k].shape, S[k].dtype)
L = np.load(r"E:\Claude code\wear\exp\v4\w25run\testD\links_oof_K7_deaug.npz")
for k in L.files:
    print("L", k, L[k].shape, L[k].dtype)
LZ = np.load(r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4\link_logodds.npz")
print([k for k in LZ.files][:12], len(LZ.files))
print(LZ["oof_0_cand"].shape, LZ["oof_0_L"][:2, :8], LZ["oof_0_cand"][:2, :8])
Q = np.load(r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4\links.npz")
for k in Q.files:
    print("Q", k, Q[k].shape)
print(np.percentile(Q["qn_ref"], [1, 10, 50, 90, 99, 99.9]))
LT = np.load(r"E:\Claude code\wear\exp\v4\w25run\testD\links_test_K7_deaug.npz")
for k in LT.files:
    print("LT", k, LT[k].shape)
