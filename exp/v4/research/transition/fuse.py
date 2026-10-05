"""Stage 1: reproduce the 2-fit late fusion of v4_combine.py (--parts K7,K9 --onehot) up to the pre-refiner labels and
cache every refiner input in this directory (no writes elsewhere)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
from v4_local import H, macro_f1, TRAIN_SETS, FOLDS, count_targets, finish_targets, log
from v4_combine import load_part, counts_for

K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fits = [load_part(K7), load_part(K9)]
    F = fits[0]
    y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64); fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    assert (fits[1]["oof_y"] == y).all() and (fits[1]["test_sbj"] == tsbj).all()
    w = np.ones(2) / 2
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(sum(wi * np.log(np.clip(M, 1e-9, None)) for wi, M in zip(w, Ms))))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    Bo = H.lsm(sum(wi * f["B2_OOF"].astype(np.float64) for wi, f in zip(w, fits)))
    Bt = H.lsm(sum(wi * f["B2_TEST"].astype(np.float64) for wi, f in zip(w, fits)))
    cnt, ct, key, kt, true = counts_for([Po, np.exp(Bo)], [Pt, np.exp(Bt)], y, sbj, tsbj, fold_of, True)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
    Qo, Qt = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    log(f"fused: OOF F1 {macro_f1(y, fin_o):.4f} per fold " + " ".join(f"{macro_f1(y[fold == f_], fin_o[fold == f_]):.4f}" for f_ in range(FOLDS)))
    lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0)); lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in fits], 0))
    oof_succ = np.concatenate([f["oof_succ"][:8] for f in fits]); oof_score = np.concatenate([f["oof_score"][:8] for f in fits])
    test_succ = np.concatenate([f["test_succ"][:8] for f in fits]); test_score = np.concatenate([f["test_score"][:8] for f in fits])
    np.savez(os.path.join(HERE, "cache.npz"), fin_o=fin_o, fin_t=fin_t, Po=Po.astype(np.float32), Pt=Pt.astype(np.float32),
             Qo=Qo.astype(np.float32), Qt=Qt.astype(np.float32), Bo=Bo.astype(np.float32), Bt=Bt.astype(np.float32),
             lwo=lwo.astype(np.float32), lwt=lwt.astype(np.float32), oof_succ=oof_succ, oof_score=oof_score,
             test_succ=test_succ, test_score=test_score, y=y, sbj=sbj, fold=fold, tsbj=tsbj,
             sensor_oof=F["sensor_oof"], sensor_test=F["sensor_test"], oof_rec=F["oof_rec"], oof_start=F["oof_start"],
             true_succ=F["true_succ"], ids=F["ids"])
    log("cached")


if __name__ == "__main__":
    main()
