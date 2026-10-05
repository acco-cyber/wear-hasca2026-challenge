"""Fused K7+K9 decode with a given count-prior file, then the v4 boundary refiner (same as v4_combine.py --onehot apart
from the counts).  python run_final.py <counts.npz> <tag> [--write]
counts.npz: cnt (OOF keys, nested), ct (test keys), key, kt  (key order = profile_features order)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
from v4_local import H, macro_f1, TRAIN_SETS, FOLDS, count_targets, finish_targets, refine, load_fit, write_sub, log, W

K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cpath, tag = sys.argv[1], sys.argv[2]; write = "--write" in sys.argv
    Z = np.load(cpath); cnt, ct, key, kt = Z["cnt"], Z["ct"], Z["key"], Z["kt"]
    fits = [load_fit(d) for d in (K7, K9)]
    for f in fits:
        f["P_o"], f["P_t"] = f["PB_OOF"].astype(np.float64), f["PB_TEST"].astype(np.float64)
    F = fits[0]; y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64)
    w = np.array([0.5, 0.5])
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(sum(wi * np.log(np.clip(M, 1e-9, None)) for wi, M in zip(w, Ms))))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    Bo = H.lsm(sum(wi * f["B2_OOF"].astype(np.float64) for wi, f in zip(w, fits))); Bt = H.lsm(sum(wi * f["B2_TEST"].astype(np.float64) for wi, f in zip(w, fits)))
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
    Qo, Qt = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    log(f"{tag}: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, fin_o):.4f} per fold "
        + " ".join(f"{macro_f1(y[fold == f_], fin_o[fold == f_]):.4f}" for f_ in range(FOLDS))
        + f"; test means {dict((int(s), round(float(ct[kt[:, 0] == s].mean()), 1)) for s in np.unique(tsbj))}")
    Lo = [(f["oof_succ"][k], f["oof_score"][k]) for f in fits for k in range(min(8, len(f["oof_succ"])))]
    Lt = [(f["test_succ"][k], f["test_score"][k]) for f in fits for k in range(min(8, len(f["test_succ"])))]
    lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0)); lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in fits], 0))
    ref_o, ref_t, _, _ = refine(fin_o, fin_t, Lo, Lt, Bo, Bt, Po, Pt, Qo, Qt, lwo, lwt, F["sensor_oof"].astype(np.int64),
                                F["sensor_test"].astype(np.int64), F["sc_o"], F["sc_t"], y, fold, 3, V.REF_THR)
    pf = [macro_f1(y[fold == f_], ref_o[fold == f_]) for f_ in range(FOLDS)]
    log(f"FINAL {tag}: OOF F1 {macro_f1(y, fin_o):.4f} -> refined {macro_f1(y, ref_o):.4f} | per fold " + " ".join(f"{v:.4f}" for v in pf)
        + "; " + "; ".join(f"agree {f['name']} {np.mean(ref_t == f['ref_test']):.4f}" for f in fits))
    np.save(os.path.join(HERE, f"res_{tag}_labo.npy"), ref_o.astype(np.int8)); np.save(os.path.join(HERE, f"res_{tag}_labt.npy"), ref_t.astype(np.int8))
    np.save(os.path.join(HERE, f"res_{tag}_fino.npy"), fin_o.astype(np.int8)); np.save(os.path.join(HERE, f"res_{tag}_fint.npy"), fin_t.astype(np.int8))
    if write:
        out = os.path.join(W, "subs", f"sub_research_{tag}")
        write_sub(F["ids"], ref_t, out + ".csv")
        np.save(out + "_labo.npy", ref_o.astype(np.int8)); np.save(out + "_labt.npy", ref_t.astype(np.int8))
        log(f"wrote {out}.csv")


if __name__ == "__main__":
    main()
