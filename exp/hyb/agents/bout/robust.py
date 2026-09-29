"""robustness: same ICM config on other upstream OOF prob sets"""
import os, glob
import numpy as np
from common import *
from bout_icm import decode, CFG_ICM

d = load_oof(); y, sbj, fold = d["y"], d["sbj"], d["fold"]
E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
for f in sorted(glob.glob(os.path.join(HB, "cv_*_oof_P.npy"))):
    P = np.load(f).astype(np.float64); P /= P.sum(1, keepdims=True)
    pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS)); Q = cal_Q(P, sbj, TRAIN_SETS)
    o = decode(P, Q, E, sbj, pred, **CFG_ICM); b = macro_f1(y, pred); a = macro_f1(y, o)
    pb, pa = per_fold(y, pred, fold), per_fold(y, o, fold)
    print(f"{os.path.basename(f):20s} base {b:.4f} icm {a:.4f} (+{a - b:.4f}) fold deltas {[round(pa[k] - pb[k], 4) for k in pb]}", flush=True)
