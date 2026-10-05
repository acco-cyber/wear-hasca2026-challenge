"""Late fusion of local decodes (same maths as v4_combine.py --onehot) with a choice of the link sets the boundary
refiner walks along. Imports v4_local / v4_combine unchanged.
  python combine_links.py --parts local:TAG:K4,local:TAG:K4 --tag NAME [--ref_links own|fused|both] [--fused NPZ]
                          [--sub PATH]   (writes PATH plus PATH_labo.npy / PATH_labt.npy when given)"""
import os, sys, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
import v4_combine as C
from v4_local import H, macro_f1, TRAIN_SETS, FOLDS, count_targets, finish_targets, refine, write_sub, log

V.REF_PARAMS["num_threads"] = 2
OUT = os.path.dirname(os.path.abspath(__file__))
FUSED = r"E:\Claude code\wear\work\v4\links_fused_s7s9.npz"


def per_fold(y, lab, fold):
    return [round(macro_f1(y[fold == f], lab[fold == f]), 4) for f in range(FOLDS)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ref_links", default="own", choices=["own", "fused", "both"]); ap.add_argument("--fused", default=FUSED)
    ap.add_argument("--members", type=int, default=8); ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sub", default=""); ap.add_argument("--no_refine", action="store_true")
    a = ap.parse_args()
    fits = [C.load_part(s) for s in a.parts.split(",")]
    F = fits[0]
    y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64); fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    for f in fits[1:]:
        assert (f["oof_y"] == y).all() and (f["test_sbj"] == tsbj).all()
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(np.mean([np.log(np.clip(M, 1e-9, None)) for M in Ms], 0)))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    Bo = H.lsm(np.mean([f["B2_OOF"].astype(np.float64) for f in fits], 0)); Bt = H.lsm(np.mean([f["B2_TEST"].astype(np.float64) for f in fits], 0))
    cnt, ct, key, kt, true = C.counts_for([Po, np.exp(Bo)], [Pt, np.exp(Bt)], y, sbj, tsbj, fold_of, True)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
    Qo, Qt = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    log(f"{a.tag}: fused count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, fin_o):.4f} per fold {per_fold(y, fin_o, fold)}")
    np.save(os.path.join(OUT, f"{a.tag}_Qo.npy"), Qo.astype(np.float32)); np.save(os.path.join(OUT, f"{a.tag}_Qt.npy"), Qt.astype(np.float32))
    ref_o, ref_t = fin_o, fin_t
    if not a.no_refine:
        Lo, Lt = [], []
        if a.ref_links in ("own", "both"):
            Lo += [(f["oof_succ"][k], f["oof_score"][k]) for f in fits for k in range(min(a.members, len(f["oof_succ"])))]
            Lt += [(f["test_succ"][k], f["test_score"][k]) for f in fits for k in range(min(a.members, len(f["test_succ"])))]
        if a.ref_links in ("fused", "both"):
            z = np.load(a.fused)
            Lo += [(z["oof_succ"][k].astype(np.int64), z["oof_score"][k].astype(np.float32)) for k in range(len(z["oof_succ"]))]
            Lt += [(z["test_succ"][k].astype(np.int64), z["test_score"][k].astype(np.float32)) for k in range(len(z["test_succ"]))]
        lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in fits], 0)); lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in fits], 0))
        ref_o, ref_t, _, _ = refine(fin_o, fin_t, Lo, Lt, Bo, Bt, Po, Pt, Qo, Qt, lwo, lwt, F["sensor_oof"].astype(np.int64),
                                    F["sensor_test"].astype(np.int64), F["sc_o"], F["sc_t"], y, fold, a.jobs)
    np.save(os.path.join(OUT, f"{a.tag}_labo.npy"), ref_o.astype(np.int8)); np.save(os.path.join(OUT, f"{a.tag}_labt.npy"), ref_t.astype(np.int8))
    if a.sub:
        write_sub(F["ids"], ref_t, a.sub)
        b = os.path.splitext(a.sub)[0]; np.save(b + "_labo.npy", ref_o.astype(np.int8)); np.save(b + "_labt.npy", ref_t.astype(np.int8))
    agree = "; ".join(f"agree {f['name']} {np.mean(ref_t == f['ref_test']):.4f}" for f in fits if "ref_test" in f)
    log(f"FINAL {a.tag} (refiner links {a.ref_links}): OOF F1 {macro_f1(y, fin_o):.4f} {per_fold(y, fin_o, fold)} -> refined "
        f"{macro_f1(y, ref_o):.4f} {per_fold(y, ref_o, fold)}; {agree}")


if __name__ == "__main__":
    main()
