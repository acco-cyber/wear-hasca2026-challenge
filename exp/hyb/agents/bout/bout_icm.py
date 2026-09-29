"""Hard-label Potts/ICM relabelling on a per-subject video+prediction kNN graph (post-calibration decoder).
  x_i <- argmax_c [ 0.2*a*log Q_i(c) + stick*[c == init_i] + sum_j Wn_ij [x_j == c] ]   (iterate to convergence)
  W: kNN (k) on cos(video emb) + use_p * <sqrt P_i, sqrt P_j>, weights exp((s - max)/0.1), symmetrised (max), row-normalised.
python bout_icm.py oof            -> OOF macro-F1 (+ per fold) of the chosen config and neighbours
python bout_icm.py test [stick]   -> test_bout.csv from sub_gl6_p03c03_top2_055_P.npy (init = its submitted labels)"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from common import *
from graphlib import knn_topk, build_W, icm

CFG_ICM = dict(use_p=1.0, k=5, a=4.0, stick=0.1)


def decode(P, Q, E, sbj, init, use_p, k, a, stick, tau=0.1):
    lq = np.log(np.clip(Q, 1e-9, 1)); out = init.copy()
    OH = np.zeros_like(lq); OH[np.arange(len(init)), init] = 1
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); nb, v = knn_topk(E, P, ii, K=max(k, 10), use_p=use_p)
        W = build_W(nb, v, k, tau, -np.ones(len(ii), np.int64), None, 0.0)
        dg = np.asarray(W.sum(1)).ravel(); dg[dg == 0] = 1
        out[ii] = icm(sp.diags(1 / dg) @ W, 0.2 * a * lq[ii] + stick * OH[ii], a=1.0, init=init[ii], iters=10)
    return out


def main():
    mode = sys.argv[1]
    if mode == "oof":
        d = load_oof(); y, sbj, fold = d["y"], d["sbj"], d["fold"]; P = d["P"]
        pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS)); Q = cal_Q(P, sbj, TRAIN_SETS)
        E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        base = macro_f1(y, pred); print("baseline", round(base, 4), per_fold(y, pred, fold))
        o = decode(P, Q, E, sbj, pred, **CFG_ICM)
        print("ICM", CFG_ICM, f"F1 {macro_f1(y, o):.4f} (+{macro_f1(y, o) - base:.4f})", per_fold(y, o, fold), "changed", round(np.mean(o != pred), 4))
        print("null share true/base/icm:", round(np.mean(y == 0), 4), round(np.mean(pred == 0), 4), round(np.mean(o == 0), 4))
        np.save(os.path.join(OUT, "oof_icm_labels.npy"), o)
    else:
        stick = float(sys.argv[2]) if len(sys.argv) > 2 else CFG_ICM["stick"]
        bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64)
        P = np.load(r"E:\Claude code\wear\subs\sub_gl6_p03c03_top2_055_P.npy").astype(np.float64); P /= P.sum(1, keepdims=True)
        sub = pd.read_csv(r"E:\Claude code\wear\subs\sub_gl6_p03c03_top2_055.csv").sort_values("id").target_feature.to_numpy().astype(np.int64)
        Q = cal_Q(P, sbj, {})
        E = load_emb("test"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        cfg = dict(CFG_ICM); cfg["stick"] = stick
        o = decode(P, Q, E, sbj, sub, **cfg)
        ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True); assert (ids == np.arange(len(ids))).all()
        fn = os.path.join(OUT, "test_bout.csv" if stick == CFG_ICM["stick"] else f"test_bout_stick{stick}.csv")
        pd.DataFrame({"id": np.arange(len(o)), "target_feature": o.astype(int)}).to_csv(fn, index=False)
        print("wrote", fn, "cfg", cfg)
        print(f"agreement with sub_gl6_p03c03_top2_055.csv: {np.mean(o == sub):.4f} (changed {np.sum(o != sub)} rows)")
        for s in np.unique(sbj):
            m = sbj == s
            print(f"  sbj {s}: n {m.sum()} null share sub {np.mean(sub[m] == 0):.3f} -> bout {np.mean(o[m] == 0):.3f}; agree {np.mean(o[m] == sub[m]):.4f}")
        ch = o != sub
        print("changed rows: from-null", np.sum(ch & (sub == 0)), "to-null", np.sum(ch & (o == 0)), "act->act", np.sum(ch & (sub > 0) & (o > 0)))


if __name__ == "__main__":
    main()
