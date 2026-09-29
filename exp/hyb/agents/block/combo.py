"""combine cached label-free block estimates by summing log-odds (weights w)"""
import numpy as np
c = dict(np.load("oof_p1.npz"))
lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1 - 1e-4))
sig = lambda z: 1 / (1 + np.exp(-z))
c["cmb_knn_nslink"] = sig(lo(c["prop_knn_resid_a9"]) + 0.5 * lo(c["ns_link"]))
c["cmb_knn_both_ns"] = sig(0.5 * lo(c["prop_knn_resid_a9"]) + 0.5 * lo(c["prop_both_raw"]) + 0.5 * lo(c["ns_link"]))
np.savez("oof_p1.npz", **c)
