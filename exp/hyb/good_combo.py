"""CV of post-hoc combinations on the public 0.927 notebook's OOF final probabilities with OUR pipeline's OOF outputs
(L2-faithful recipe): gate with our labels, geometric blend with our calibrated probabilities.
python good_combo.py <our_oof_P.npy> <our_oof_labels.npy>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import macro_f1, KEEP, TRAIN_SETS, calibrate, CFG, _norm_rows
W = r"E:\Claude code\wear"
O = np.load(os.path.join(W, "public_src", "good927", "out", "final_probabilities.npz"))["oof"].astype(np.float64)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}; y, sbj, fold = sm["y"], sm["sbj"], sm["fold"]
P = np.load(sys.argv[1]).astype(np.float64); P /= P.sum(1, keepdims=True); ours = np.load(sys.argv[2]).astype(np.int64)
base = O.argmax(1); conf = O.max(1); srt = np.argsort(-O, 1)
print(f"theirs {macro_f1(y, base):.4f} | ours labels {macro_f1(y, ours):.4f} | agreement {np.mean(base == ours):.4f}")
dis = base != ours; print(f"disagreements {dis.mean():.4f}: theirs right {np.mean(base[dis] == y[dis]):.3f}, ours right {np.mean(ours[dis] == y[dis]):.3f}")
for tau in (0.5, 0.6, 0.7, 0.8):
    for rule in ("plain", "top2"):
        g = conf < tau
        if rule == "top2":
            g = g & ((srt[:, 0] == ours) | (srt[:, 1] == ours))
        lab = base.copy(); lab[g] = ours[g]
        print(f"  gate {tau} {rule}: F1 {macro_f1(y, lab):.4f} gated {g.mean():.3f} | per fold " + " ".join(f"{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(5)))
Q = np.clip(P, 1e-12, None) ** 2; Q = calibrate(Q / Q.sum(1, keepdims=True), sbj, TRAIN_SETS, CFG["per_ex"], CFG["null_min"])
for w in (0.1, 0.2, 0.3):
    B = _norm_rows(O ** (1 - w) * Q ** w); print(f"  geometric blend w={w}: F1 {macro_f1(y, B.argmax(1)):.4f}")
