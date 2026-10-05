"""Exploratory probe (not a reported score): feed the outer-fold specialist null probability back into the fused P before
the count-constrained Sinkhorn, so flips have to respect the learned per-subject counts. Pre-refiner F1 only."""
import os, sys
import numpy as np
from ne_common import HERE, K7, K9
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_combine as C
from v4_local import macro_f1, TRAIN_SETS, FOLDS, count_targets, finish_targets, CFG

tag = sys.argv[1] if len(sys.argv) > 1 else "null_cand3"
r = np.load(os.path.join(HERE, f"res_{tag}.npz")); b = np.load(os.path.join(HERE, "base_cache.npz"))
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
y = st["oof_y"].astype(np.int64); sbj = st["oof_sbj"].astype(np.int64); fold = st["oof_fold"].astype(np.int64); tsbj = st["test_sbj"].astype(np.int64)
fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
Po, Pt, Qo = b["Po"].astype(np.float64), b["Pt"].astype(np.float64), b["Qo"].astype(np.float64)
cnt, ct, key, kt, true = C.counts_for([Po, np.exp(b["Bo"].astype(np.float64))], [Pt, np.exp(b["Bt"].astype(np.float64))], y, sbj, tsbj, fold_of, True)
tg = count_targets(sbj, TRAIN_SETS, key, cnt)
Q0 = finish_targets(Po, sbj, tg); fin = Q0.argmax(1)
print("check base fin", macro_f1(y, fin), np.abs(Q0 - Qo).max())
idx, p = r["idx"], np.clip(r["p_outer"], 1e-4, 1 - 1e-4)
lg = lambda x: np.log(x) - np.log1p(-x)
q0 = np.clip(Q0[idx, 0], 1e-6, 1 - 1e-6)
T = CFG["sharpen_T"]
for w in (0.25, 0.5, 0.75, 1.0, 1.5):
    P2 = Po.copy(); P2[idx, 0] *= np.exp(T * w * (lg(p) - lg(q0)))
    lab = finish_targets(P2 / P2.sum(1, keepdims=True), sbj, tg).argmax(1)
    print(f"w {w}: pre-refiner F1 {macro_f1(y, fin):.4f} -> {macro_f1(y, lab):.4f}; per fold "
          + " ".join(f"{macro_f1(y[fold == k], fin[fold == k]):.4f}->{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(FOLDS))
          + f"; changed {(lab != fin).sum()}; vs refined base {macro_f1(y, b['ref_o']):.4f}")
