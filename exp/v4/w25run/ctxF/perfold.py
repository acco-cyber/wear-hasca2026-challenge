"""per-fold OOF F1 of a decode's labels vs the K7/K9 baselines (kernel final QB_OOF = pre-refiner, ref_oof = refined)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
fit = sys.argv[1]; tags = sys.argv[2:]
st = np.load(os.path.join({"K7": K7, "K9": K9}[fit], "stage.npz"), allow_pickle=True)
y = st["oof_y"].astype(np.int64); fold = st["oof_fold"].astype(np.int64)
pf = lambda lab: [macro_f1(y[fold == f], lab[fold == f]) for f in range(5)]
pre0, ref0 = st["QB_OOF"].argmax(1), st["ref_oof"].astype(np.int64)
print(f"{fit} baseline pre {macro_f1(y, pre0):.4f} " + " ".join(f"{v:.4f}" for v in pf(pre0)) + f" | refined {macro_f1(y, ref0):.4f} " + " ".join(f"{v:.4f}" for v in pf(ref0)))
for t in tags:
    p = os.path.join(W, "subs", f"sub_v4l_{t}")
    if not os.path.exists(p + "_labo.npy"):
        continue
    pre = np.load(p + "_Qo.npy").argmax(1); ref = np.load(p + "_labo.npy").astype(np.int64)
    up_pre = sum(a > b for a, b in zip(pf(pre), pf(pre0))); up_ref = sum(a > b for a, b in zip(pf(ref), pf(ref0)))
    print(f"{t}: pre {macro_f1(y, pre):.4f} ({macro_f1(y, pre) - macro_f1(y, pre0):+.4f}, {up_pre}/5 up) " + " ".join(f"{v:.4f}" for v in pf(pre))
          + f" | refined {macro_f1(y, ref):.4f} ({macro_f1(y, ref) - macro_f1(y, ref0):+.4f}, {up_ref}/5 up) " + " ".join(f"{v:.4f}" for v in pf(ref)))
