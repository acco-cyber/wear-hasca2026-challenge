"""Reproduce the fused K7+K9 decode (v4_combine --parts K7,K9 --onehot, before the refiner) and cache every input the
boundary refiner needs, plus per-fit extras (each fit's final P, own-count labels, blend) for the refiner research.
  python prep.py"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_combine as C
from v4_local import H, macro_f1, TRAIN_SETS, FOLDS, count_targets, finish_targets, log, W

K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def main():
    os.makedirs(OUT, exist_ok=True)
    fits = [C.load_part(K7), C.load_part(K9)]
    F = fits[0]
    y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64); fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    w = np.ones(2) / 2
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(sum(wi * np.log(np.clip(M, 1e-9, None)) for wi, M in zip(w, Ms))))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    Bo = H.lsm(sum(wi * f["B2_OOF"].astype(np.float64) for wi, f in zip(w, fits)))
    Bt = H.lsm(sum(wi * f["B2_TEST"].astype(np.float64) for wi, f in zip(w, fits)))
    Bpo, Bpt = np.exp(Bo), np.exp(Bt)
    per = {}
    for i, f in enumerate(fits):
        bo, bt = np.exp(f["B2_OOF"].astype(np.float64)), np.exp(f["B2_TEST"].astype(np.float64))
        c_o, c_t, key, kt, true = C.counts_for([f["P_o"], bo], [f["P_t"], bt], y, sbj, tsbj, fold_of, True)
        qo = finish_targets(f["P_o"], sbj, count_targets(sbj, TRAIN_SETS, key, c_o))
        qt = finish_targets(f["P_t"], tsbj, count_targets(tsbj, {}, kt, c_t))
        log(f"fit {f['name']}: own-count OOF F1 {macro_f1(y, qo.argmax(1)):.4f}, kernel refined {macro_f1(y, f['ref_oof']):.4f}")
        per.update({f"f{i}_Po": f["P_o"].astype(np.float32), f"f{i}_Pt": f["P_t"].astype(np.float32),
                    f"f{i}_Qo": qo.astype(np.float32), f"f{i}_Qt": qt.astype(np.float32),
                    f"f{i}_Bo": f["B2_OOF"].astype(np.float32), f"f{i}_Bt": f["B2_TEST"].astype(np.float32),
                    f"f{i}_refo": f["ref_oof"].astype(np.int16), f"f{i}_reft": f["ref_test"].astype(np.int16),
                    f"f{i}_QBo": f["QB_OOF"].astype(np.float32), f"f{i}_QBt": f["QB_TEST"].astype(np.float32),
                    f"f{i}_lwo": f["oof_logp"].astype(np.float32), f"f{i}_lwt": f["test_logp"].astype(np.float32)})
    cnt, ct, key, kt, true = C.counts_for([Po, Bpo], [Pt, Bpt], y, sbj, tsbj, fold_of, True)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
    Qo, Qt = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    log(f"fused: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, fin_o):.4f} per fold "
        + " ".join(f"{macro_f1(y[fold == f_], fin_o[fold == f_]):.4f}" for f_ in range(FOLDS)))
    ref = os.path.join(W, "subs", "sub_v4c_b2_s7s9_Qo.npy")
    if os.path.exists(ref):
        log(f"agreement with saved sub_v4c_b2_s7s9 Qo argmax: {np.mean(np.load(ref).argmax(1) == fin_o):.5f}")
    lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0))
    lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in fits], 0))
    Lo_s = np.stack([f["oof_succ"][k] for f in fits for k in range(8)]); Lo_c = np.stack([f["oof_score"][k] for f in fits for k in range(8)])
    Lt_s = np.stack([f["test_succ"][k] for f in fits for k in range(8)]); Lt_c = np.stack([f["test_score"][k] for f in fits for k in range(8)])
    np.savez(os.path.join(OUT, "fused.npz"), y=y, sbj=sbj, fold=fold, tsbj=tsbj, ids=F["ids"], true_succ=F["true_succ"],
             Po=Po.astype(np.float32), Pt=Pt.astype(np.float32), Bo=Bo.astype(np.float32), Bt=Bt.astype(np.float32),
             Qo=Qo.astype(np.float32), Qt=Qt.astype(np.float32), fin_o=fin_o, fin_t=fin_t,
             lwo=lwo.astype(np.float32), lwt=lwt.astype(np.float32), sens_o=F["sensor_oof"], sens_t=F["sensor_test"],
             Lo_s=Lo_s, Lo_c=Lo_c, Lt_s=Lt_s, Lt_c=Lt_c, **per)
    eo, po, vo, mo = F["sc_o"]; et, pt, vt, mt = F["sc_t"]
    np.savez(os.path.join(OUT, "scalars.npz"), ener_o=eo, post_o=po, vmot_o=vo, vmean_o=mo.astype(np.float32),
             ener_t=et, post_t=pt, vmot_t=vt, vmean_t=mt.astype(np.float32))
    log("cached")


if __name__ == "__main__":
    main()
