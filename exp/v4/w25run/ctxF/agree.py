"""change rates vs the fit's own kernel output: OOF (labo vs ref_oof) and test (labt vs ref_test)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
fit = sys.argv[1]; st = np.load(os.path.join({"K7": K7, "K9": K9}[fit], "stage.npz"), allow_pickle=True)
for t in sys.argv[2:]:
    p = os.path.join(W, "subs", f"sub_v4l_{t}")
    if os.path.exists(p + "_labo.npy"):
        lo, lt = np.load(p + "_labo.npy"), np.load(p + "_labt.npy")
        print(f"{t}: OOF changed vs ref_oof {np.mean(lo != st['ref_oof']):.4f} | test changed vs ref_test {np.mean(lt != st['ref_test']):.4f}")
