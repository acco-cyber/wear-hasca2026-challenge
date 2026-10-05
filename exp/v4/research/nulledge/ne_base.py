"""Reproduce the 2-fit late-fusion baseline (v4_combine --parts K7,K9 --onehot) with the v4_combine / v4_local functions
and cache every array the null-edge specialist needs (fused P/Q, blends, labels before/after the refiner, refiner
row probabilities)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import numpy as np
import v4_local as V
import v4_combine as C
from v4_local import H, macro_f1, N_CLS, TRAIN_SETS, FOLDS, count_targets, finish_targets, refine, log

V.REF_PARAMS["num_threads"] = 2
K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_cache.npz")


def main():
    fits = [C.load_part(K7), C.load_part(K9)]
    w = np.ones(2) / 2
    F = fits[0]
    y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64); fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    assert (fits[1]["oof_y"] == y).all() and (fits[1]["test_sbj"] == tsbj).all()
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(sum(wi * np.log(np.clip(M, 1e-9, None)) for wi, M in zip(w, Ms))))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    Bo = H.lsm(sum(wi * f["B2_OOF"].astype(np.float64) for wi, f in zip(w, fits)))
    Bt = H.lsm(sum(wi * f["B2_TEST"].astype(np.float64) for wi, f in zip(w, fits)))
    Bpo, Bpt = np.exp(Bo), np.exp(Bt)
    cnt, ct, key, kt, true = C.counts_for([Po, Bpo], [Pt, Bpt], y, sbj, tsbj, fold_of, True)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
    Qo, Qt = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    log(f"fused: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, fin_o):.4f} per fold "
        + " ".join(f"{macro_f1(y[fold == f_], fin_o[fold == f_]):.4f}" for f_ in range(FOLDS)))
    Lo = [(f["oof_succ"][k], f["oof_score"][k]) for f in fits for k in range(min(8, len(f["oof_succ"])))]
    Lt = [(f["test_succ"][k], f["test_score"][k]) for f in fits for k in range(min(8, len(f["test_succ"])))]
    lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0))
    lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in fits], 0))
    ref_o, ref_t, ro, rt = refine(fin_o, fin_t, Lo, Lt, Bo, Bt, Po, Pt, Qo, Qt, lwo, lwt, F["sensor_oof"].astype(np.int64),
                                  F["sensor_test"].astype(np.int64), F["sc_o"], F["sc_t"], y, fold, 3, V.REF_THR)
    log(f"BASE: OOF F1 {macro_f1(y, fin_o):.4f} -> refined {macro_f1(y, ref_o):.4f} per fold "
        + " ".join(f"{macro_f1(y[fold == f_], ref_o[fold == f_]):.4f}" for f_ in range(FOLDS)))
    np.savez_compressed(OUT, Po=Po.astype(np.float32), Pt=Pt.astype(np.float32), Qo=Qo.astype(np.float32), Qt=Qt.astype(np.float32),
                        Bo=Bo.astype(np.float32), Bt=Bt.astype(np.float32), lwo=lwo.astype(np.float32), lwt=lwt.astype(np.float32),
                        fin_o=fin_o, fin_t=fin_t, ref_o=ref_o, ref_t=ref_t,
                        r_tl=ro[0], r_ot=ro[1], r_pr=ro[2], rt_tl=rt[0], rt_ot=rt[1], rt_pr=rt[2],
                        P7o=fits[0]["P_o"].astype(np.float32), P7t=fits[0]["P_t"].astype(np.float32),
                        P9o=fits[1]["P_o"].astype(np.float32), P9t=fits[1]["P_t"].astype(np.float32))
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
