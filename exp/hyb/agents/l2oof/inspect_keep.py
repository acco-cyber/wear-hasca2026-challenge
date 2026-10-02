import numpy as np, os
K = r"E:\Claude code\wear\work\hanbat\keep"
sm = dict(np.load(os.path.join(K, "sim_meta.npz")))
print({k: (v.shape, v.dtype) for k, v in sm.items()})
fold_of = {int(s): int(f) for s, f in zip(sm["sbj"], sm["fold"])}
print("fold_of", dict(sorted(fold_of.items())))
bl = np.load(os.path.join(K, "blend.npz"))
print({k: (bl[k].shape, bl[k].dtype) for k in bl.files})
l0 = np.load(os.path.join(K, "links_L0.npz")); print({k: (l0[k].shape, l0[k].dtype) for k in l0.files})
l2 = np.load(os.path.join(K, "links_L2_test.npz")); print({k: (l2[k].shape, l2[k].dtype) for k in l2.files})
for s in range(1, 11):
    p = rf"E:\Claude code\wear\data\train\videomae_feat\sbj_{s}.npy"
    try:
        v = np.load(p, mmap_mode="r"); print(s, v.shape, v.dtype)
    except Exception as e:
        print(s, "ERR", e)
import hashlib
sim = {k: v.astype(np.int64) for k, v in sm.items()}
print(hashlib.sha1(sim["sensor"].tobytes()).hexdigest()[:12], hashlib.sha1(np.stack([sim["rec"], sim["start"]]).tobytes()).hexdigest()[:12])
# L0 diag
key = {(r, s): n for n, (r, s) in enumerate(zip(sim["rec"].tolist(), sim["start"].tolist()))}
ts = np.array([key.get((r, s + 50), -1) for r, s in zip(sim["rec"].tolist(), sim["start"].tolist())])
su = l0["oof_succ"]; m = su >= 0
print("L0 linked", m.mean(), "exact", (su[m] == ts[m]).mean(), "same", (sim["y"][su[m]] == sim["y"][m]).mean())
