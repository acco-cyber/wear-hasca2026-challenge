import numpy as np
for d in [r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"]:
    st = np.load(d + r"\stage.npz", allow_pickle=True)
    for k in st.files:
        print(k, st[k].shape, st[k].dtype)
    lk = np.load(d + r"\links.npz")
    for k in lk.files:
        print(k, lk[k].shape, lk[k].dtype)
    z = np.load(d + r"\tile_scalars.npz")
    for k in z.files:
        print(k, z[k].shape, z[k].dtype)
    z = np.load(d + r"\dec_cache.npz", allow_pickle=True)
    for k in z.files:
        print('dec', k, z[k].shape, z[k].dtype)
    y = st["oof_y"]; print(np.bincount(y)); print(np.unique(st["oof_sbj"]), np.unique(st["oof_fold"]))
    print({int(s): int(f) for s, f in zip(st["oof_sbj"], st["oof_fold"])})
    print(np.unique(st["test_sbj"], return_counts=True))
