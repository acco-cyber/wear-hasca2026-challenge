import os
import numpy as np
S = r"E:\Claude code\wear\subs"
for a, b in (("asmT_K7", "asmB_base"), ("asmT_K9", "asmT_K9base")):
    for k in ("labt", "labo"):
        x = np.load(os.path.join(S, f"sub_v4l_{a}_{k}.npy")); y = np.load(os.path.join(S, f"sub_v4l_{b}_{k}.npy"))
        print(a, "vs", b, k, "shape", x.shape, "differ", int((x != y).sum()), f"({np.mean(x != y):.4f})")
