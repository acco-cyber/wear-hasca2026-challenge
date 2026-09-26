"""Scores for the UEC Inertial-LightGBM port.
manifest: OOF macro-F1 over the concatenated 5 contiguous-subject folds (repo's cv protocol), per-fold F1, mean.
std:      builds std_cv/oof.npy (69326,4,19) and scores it with exp/base/common.eval_single (one random valid limb per
          1-s tile, seed 5, purity >= 0.8).
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\uec\gbdt")
from train_uec import load_all, HERE

X, M = load_all()
del X
from sklearn.metrics import f1_score

out = {}
fd = sorted(glob.glob(os.path.join(HERE, "manifest_cv", "fold_*", "done.json")))
if fd:
    idx, P, per = [], [], []
    for p in fd:
        d = os.path.dirname(p)
        r = json.load(open(p))
        idx.append(np.load(os.path.join(d, "val_idx.npy")))
        P.append(np.load(os.path.join(d, "val_prob.npy")))
        per.append((os.path.basename(d), r["val_subjects"], r["best_iter"], round(r["val_macro_f1"], 4)))
    idx = np.concatenate(idx)
    P = np.concatenate(P)
    y = M["y"][idx]
    out["manifest_oof_macro_f1"] = float(f1_score(y, P.argmax(1), average="macro"))
    out["manifest_fold_mean_f1"] = float(np.mean([p[3] for p in per]))
    out["manifest_folds"] = per
    out["manifest_n_folds_done"] = len(fd)
    # per-sensor-group OOF (repo exports arm/leg bundles)
    sens = M["sensor"][idx]
    for g, ks in (("arm", ("ra", "la")), ("leg", ("rl", "ll"))):
        m = np.isin(sens, ks)
        out[f"manifest_oof_{g}_f1"] = float(f1_score(y[m], P[m].argmax(1), average="macro"))

fs = sorted(glob.glob(os.path.join(HERE, "std_cv", "fold_*", "done.json")))
if len(fs) == 5:
    import pandas as pd
    meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
    key = {(s, int(t)): i for i, (s, t) in enumerate(zip(meta.session.to_numpy(), meta.t.to_numpy()))}
    LIMB = {"la": 0, "ll": 1, "ra": 2, "rl": 3}   # LIMBS = left_arm, left_leg, right_arm, right_leg
    oof = np.full((len(meta), 4, 19), np.nan, np.float32)
    hit = miss = 0
    for p in fs:
        d = os.path.dirname(p)
        ei = np.load(os.path.join(d, "extra_idx.npy"))
        EP = np.load(os.path.join(d, "extra_prob.npy"))
        for r, pr in zip(ei, EP):
            k = (M["session"][r], int(M["start"][r]) // 50)
            i = key.get(k)
            if i is None:
                miss += 1
                continue
            oof[i, LIMB[M["sensor"][r]]] = pr
            hit += 1
    np.save(os.path.join(HERE, "std_cv", "oof.npy"), oof)
    sys.path.insert(0, r"E:\Claude code\wear\exp\base")
    import common
    out["std_single_limb_f1"] = float(common.eval_single(oof, name="uec_gbdt"))
    out["std_rows_mapped"] = hit
    out["std_rows_outside_meta"] = miss
    out["std_folds"] = [json.load(open(p)) for p in fs]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "scores.json"), "w"), indent=1)
