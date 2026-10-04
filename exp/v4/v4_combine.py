"""Late fusion of several independently decoded fits of the v4 pipeline: geometric mean of every fit's final graph
probabilities (pre-calibration P of the last pass), the learned count prior re-estimated on the fused P (profile
features from the fused P, the averaged blend and optionally every single fit), Sinkhorn with those counts, then the
boundary refiner along the matchings of all fits.
A part is either a keep4 directory of a Kaggle fork (uses PB_OOF / PB_TEST) or local:<tag>:<dir> for a local decode
(subs/sub_v4l_<tag>_Po.npy / _Pt.npy, with the blend / links of <dir>).
  python v4_combine.py --parts P1,P2[,...] --tag NAME [--w 1,1] [--fit_prof] [--no_refine] [--members 8]"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_local as V
from v4_local import (H, macro_f1, N_CLS, TRAIN_SETS, FOLDS, profile_features, fit_counts, count_targets, finish_targets,
                      refine, load_fit, write_sub, log, W)


def load_part(spec):
    if spec.startswith("q:"):          # q:<final_probabilities.npz>[:name] -- calibrated Q only; P ~ Q ** sharpen_T
        bits = spec.split(":"); pth = ":".join(bits[1:3]) if len(bits[1]) == 1 else bits[1]   # allow drive letters
        nm = bits[-1] if len(bits) > (3 if len(bits[1]) == 1 else 2) else os.path.basename(os.path.dirname(pth))
        z = np.load(pth); T = 0.5                     # the kernel's sharpen_T (inverts its finish)
        prox = lambda Q: (lambda R: R / R.sum(1, keepdims=True))(np.clip(Q.astype(np.float64), 1e-12, None) ** T)
        return {"P_o": prox(z["oof"]), "P_t": prox(z["test"]), "name": nm, "q_only": True,
                **({"ref_test": z["test_labels"], "ref_oof": z["oof_labels"], "QB_OOF": z["oof"]} if "test_labels" in z.files else {})}
    if spec.startswith("local:"):
        _, tag, d = spec.split(":", 2)
        f = load_fit(d); p = os.path.join(W, "subs", f"sub_v4l_{tag}")
        f["P_o"], f["P_t"] = np.load(p + "_Po.npy").astype(np.float64), np.load(p + "_Pt.npy").astype(np.float64)
        f["name"] = tag
    else:
        f = load_fit(spec); f["P_o"], f["P_t"] = f["PB_OOF"].astype(np.float64), f["PB_TEST"].astype(np.float64)
    return f


def counts_for(mats_o, mats_t, y, sbj, tsbj, fold_of, onehot=False):
    X, key = profile_features(mats_o, sbj, TRAIN_SETS); Xt, kt = profile_features(mats_t, tsbj, {})
    if onehot:
        X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1); Xt = np.concatenate([Xt, np.eye(N_CLS - 1)[kt[:, 1] - 1]], 1)
    true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64)
    kf = np.array([fold_of[int(s)] for s, _ in key]); cnt = np.zeros(len(true))
    for f_ in range(FOLDS):
        cnt[kf == f_] = fit_counts(X[kf != f_], true[kf != f_], X[kf == f_])
    return cnt, fit_counts(X, true, Xt), key, kt, true


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--w", default=""); ap.add_argument("--fit_prof", action="store_true"); ap.add_argument("--onehot", action="store_true")
    ap.add_argument("--count_avg", action="store_true", help="average the joint count prediction with every fit's own prediction")
    ap.add_argument("--no_refine", action="store_true"); ap.add_argument("--members", type=int, default=8); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ref_thr", type=float, default=V.REF_THR)
    ap.add_argument("--aux", default="", help="keep4 dir to borrow tile scalars + sensors from (for parts without them)")
    ap.add_argument("--sharpen", type=float, default=0.0, help="override CFG sharpen_T for the fused finish")
    ap.add_argument("--redecode", type=int, default=0, help="extra rounds: re-decode every full fit with the fused counts")
    ap.add_argument("--whiten", type=float, default=0.5, help="per-subject whitening power used by --redecode (0 = raw)")
    a = ap.parse_args()
    if a.sharpen > 0:
        V.CFG["sharpen_T"] = a.sharpen
    fits = [load_part(s) for s in a.parts.split(",")]
    w = np.array([float(x) for x in a.w.split(",")]) if a.w else np.ones(len(fits)); w = w / w.sum()
    full = [f for f in fits if not f.get("q_only")]
    F = full[0]
    if a.aux and "sc_o" not in F:               # tile scalars / sensors are deterministic: borrow them from a keep4 run
        X_ = load_fit(a.aux); assert (X_["oof_y"] == F["oof_y"]).all() and (X_["test_sbj"] == F["test_sbj"]).all()
        F.update(sc_o=X_["sc_o"], sc_t=X_["sc_t"], sensor_oof=X_["sensor_oof"], sensor_test=X_["sensor_test"])
    y = F["oof_y"].astype(np.int64); sbj = F["oof_sbj"].astype(np.int64); fold = F["oof_fold"].astype(np.int64)
    tsbj = F["test_sbj"].astype(np.int64); fold_of = {int(s): int(f_) for s, f_ in zip(sbj, fold)}
    for f in full[1:]:
        assert (f["oof_y"] == y).all() and (f["test_sbj"] == tsbj).all()
    for f in fits:
        assert f["P_o"].shape == (len(y), N_CLS) and f["P_t"].shape == (len(tsbj), N_CLS)
    geo = lambda Ms: (lambda G: G / G.sum(1, keepdims=True))(np.exp(sum(wi * np.log(np.clip(M, 1e-9, None)) for wi, M in zip(w, Ms))))
    Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
    wf = np.array([wi for wi, f in zip(w, fits) if not f.get("q_only")]); wf = wf / wf.sum()
    Bo = H.lsm(sum(wi * f["B2_OOF"].astype(np.float64) for wi, f in zip(wf, full))); Bt = H.lsm(sum(wi * f["B2_TEST"].astype(np.float64) for wi, f in zip(wf, full)))
    Bpo, Bpt = np.exp(Bo), np.exp(Bt)
    # single fits with their own counts, for reference
    for f in fits:
        bo = np.exp(f["B2_OOF"].astype(np.float64)) if "B2_OOF" in f else Bpo; bt = np.exp(f["B2_TEST"].astype(np.float64)) if "B2_TEST" in f else Bpt
        c_o, c_t, key, kt, true = counts_for([f["P_o"], bo], [f["P_t"], bt], y, sbj, tsbj, fold_of, a.onehot)
        f["cnt"], f["ct"] = c_o, c_t
        lab = finish_targets(f["P_o"], sbj, count_targets(sbj, TRAIN_SETS, key, c_o)).argmax(1)
        log(f"  {f['name']}: count error {np.abs(c_o - true).mean():.2f}, OOF F1 {macro_f1(y, lab):.4f}"
            + (f" (kernel final {macro_f1(y, f['QB_OOF'].argmax(1)):.4f}, refined {macro_f1(y, f['ref_oof']):.4f})" if "ref_oof" in f else ""))
    xo = [f["P_o"] for f in fits] if a.fit_prof else []; xt = [f["P_t"] for f in fits] if a.fit_prof else []
    cnt, ct, key, kt, true = counts_for([Po, Bpo] + xo, [Pt, Bpt] + xt, y, sbj, tsbj, fold_of, a.onehot)
    if a.count_avg:
        cnt = 0.5 * cnt + 0.5 * np.mean([f["cnt"] for f in fits], 0); ct = 0.5 * ct + 0.5 * np.mean([f["ct"] for f in fits], 0)
    tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
    for r in range(a.redecode):                  # every full fit decoded again with the FUSED count targets, re-fused
        cfg_names = V.make_cfgs(1.5, 0.3 if a.whiten > 0 else None)
        for f in fits:
            if f.get("q_only"):
                continue
            eo = V.whitened_subject(f["oof_emb"], sbj, a.whiten, 0.1) if a.whiten > 0 else f["oof_emb"]
            et = V.whitened_subject(f["test_emb"], tsbj, a.whiten, 0.1) if a.whiten > 0 else f["test_emb"]
            Lo_f = [(f["oof_succ"][k], f["oof_score"][k]) for k in range(min(a.members, len(f["oof_succ"])))]
            Lt_f = [(f["test_succ"][k], f["test_score"][k]) for k in range(min(a.members, len(f["test_succ"])))]
            f["P_o"], _ = V.bag_P(dict(logp=f["B2_OOF"].astype(np.float32), emb=eo, grp=sbj, sbj=sbj, sets=TRAIN_SETS), Lo_f, tg, cfg_names, a.jobs)
            f["P_t"], _ = V.bag_P(dict(logp=f["B2_TEST"].astype(np.float32), emb=et, grp=tsbj, sbj=tsbj, sets={}), Lt_f, tgt, cfg_names, a.jobs)
            log(f"  redecode {r + 1}: {f['name']} done")
        Po, Pt = geo([f["P_o"] for f in fits]), geo([f["P_t"] for f in fits])
        xo = [f["P_o"] for f in fits] if a.fit_prof else []; xt = [f["P_t"] for f in fits] if a.fit_prof else []
        cnt, ct, key, kt, true = counts_for([Po, Bpo] + xo, [Pt, Bpt] + xt, y, sbj, tsbj, fold_of, a.onehot)
        tg, tgt = count_targets(sbj, TRAIN_SETS, key, cnt), count_targets(tsbj, {}, kt, ct)
        log(f"redecode {r + 1}: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, finish_targets(Po, sbj, tg).argmax(1)):.4f}")
    Qo, Qt = finish_targets(Po, sbj, tg), finish_targets(Pt, tsbj, tgt)
    fin_o, fin_t = Qo.argmax(1), Qt.argmax(1)
    log(f"fused ({len(fits)} fits, w={np.round(w, 3).tolist()}): count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, fin_o):.4f} per fold "
        + " ".join(f"{macro_f1(y[fold == f_], fin_o[fold == f_]):.4f}" for f_ in range(FOLDS))
        + f"; test means {dict((int(s), round(float(ct[kt[:, 0] == s].mean()), 1)) for s in np.unique(tsbj))}")
    out = os.path.join(W, "subs", f"sub_v4c_{a.tag}")
    np.save(out + "_Qo.npy", Qo.astype(np.float32)); np.save(out + "_Qt.npy", Qt.astype(np.float32))
    ref_o, ref_t = fin_o, fin_t
    if not a.no_refine and "sc_o" in F:
        Lo = [(f["oof_succ"][k], f["oof_score"][k]) for f in full for k in range(min(a.members, len(f["oof_succ"])))]
        Lt = [(f["test_succ"][k], f["test_score"][k]) for f in full for k in range(min(a.members, len(f["test_succ"])))]
        lwo = H.lsm(np.mean([f["oof_logp"].astype(np.float64) for f in full], 0)); lwt = H.lsm(np.mean([f["test_logp"].astype(np.float64) for f in full], 0))
        ref_o, ref_t, _, _ = refine(fin_o, fin_t, Lo, Lt, Bo, Bt, Po, Pt, Qo, Qt, lwo, lwt, F["sensor_oof"].astype(np.int64),
                                    F["sensor_test"].astype(np.int64), F["sc_o"], F["sc_t"], y, fold, a.jobs, a.ref_thr)
    np.save(out + "_labo.npy", ref_o.astype(np.int8)); np.save(out + "_labt.npy", ref_t.astype(np.int8))
    write_sub(F["ids"], ref_t, out + ".csv")
    msg = "; ".join(f"agree {f['name']} {np.mean(ref_t == f['ref_test']):.4f}" for f in fits if "ref_test" in f)
    log(f"FINAL {a.tag}: OOF F1 {macro_f1(y, fin_o):.4f} -> refined {macro_f1(y, ref_o):.4f}; wrote {out}.csv; {msg}")


if __name__ == "__main__":
    main()
