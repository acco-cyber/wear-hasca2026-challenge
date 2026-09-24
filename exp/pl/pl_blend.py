"""Blend a retrained v3b-like model with lgbm_v1 and fusion_v1 (0.5/0.3/0.2, same as bl_v3b_v1_f), OOF and test.
python pl_blend.py <tag>  -> exp/pl/<tag>/bl_oof.npy, bl_test.npy"""
import os, sys, numpy as np
W = r"E:\Claude code\wear\work"; PL = r"E:\Claude code\wear\exp\pl"
tag = sys.argv[1]; ws = (0.5, 0.3, 0.2)
def lg(p): return np.log(np.clip(np.load(p), 1e-6, 1))
for kind in ("oof", "test"):
    L = ws[0] * lg(os.path.join(PL, tag, f"{kind}.npy")) + ws[1] * lg(os.path.join(W, "lgbm_v1", f"{kind}.npy")) + ws[2] * lg(os.path.join(W, "fusion_v1", f"{kind}.npy"))
    L = L - np.nanmax(L, -1, keepdims=True); P = np.exp(L); P = P / np.nansum(P, -1, keepdims=True)
    np.save(os.path.join(PL, tag, f"bl_{kind}.npy"), P.astype(np.float32)); print("saved", kind, P.shape)
