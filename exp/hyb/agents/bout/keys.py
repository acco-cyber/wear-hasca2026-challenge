import numpy as np, glob, os
for f in [r"E:\Claude code\wear\work\hanbat\keep\links_L0.npz", r"E:\Claude code\wear\work\hanbat\keep\links_L2_test.npz",
          r"E:\Claude code\wear\work\hanbat\keep\blend.npz", r"E:\Claude code\wear\work\hanbat\keep\feat_oof.npz"]:
    z = np.load(f, allow_pickle=True); print(os.path.basename(f), {k: (z[k].shape, z[k].dtype) for k in z.files})
for f in glob.glob(r"E:\Claude code\wear\work\**\*link*", recursive=True)[:40]:
    print(f, os.path.getsize(f))
