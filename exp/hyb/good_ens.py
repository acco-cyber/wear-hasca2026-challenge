"""Ensemble of independent fits of the "Learned Links + Counts" pipeline at the level of final (calibrated)
probabilities: geometric mean, optional re-Sinkhorn to the members' mean per-subject class masses.
  python good_ens.py cv   name=path_to_oof.npy[:w] ...          -> OOF macro-F1 of each member and of the ensemble
  python good_ens.py test <out.csv> name=path_to_test.npy[:w] ... [--resink]
A path may be 'npz_file::key' to read an array from an npz."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import macro_f1, KEEP
from graph_lab import calibrate_targets


def load(spec):
    nm, rest = spec.split("=", 1); w = 1.0       # name=path[@weight], path may be 'file.npz::key'
    if "@" in rest:
        rest, w_ = rest.rsplit("@", 1); w = float(w_)
    if "::" in rest:
        f, k = rest.split("::"); A = np.load(f)[k]
    else:
        A = np.load(rest)
    A = A.astype(np.float64); return nm, A / A.sum(1, keepdims=True), w


def ens(members, sbj, resink):
    ws = np.array([w for _, _, w in members]); ws = ws / ws.sum()
    G = np.exp(sum(w * np.log(np.clip(A, 1e-9, None)) for (_, A, _), w in zip(members, ws))); G /= G.sum(1, keepdims=True)
    if resink:
        T = {int(s): sum(w * A[sbj == s].sum(0) for (_, A, _), w in zip(members, ws)) for s in np.unique(sbj)}
        G = calibrate_targets(G, sbj, T)
    return G


mode = sys.argv[1]; resink = "--resink" in sys.argv; args = [a for a in sys.argv[2:] if a != "--resink"]
if mode == "cv":
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}; y, sbj, fold = sm["y"], sm["sbj"], sm["fold"]
    members = [load(a) for a in args]
    for nm, A, w in members:
        print(f"{nm:14s} F1 {macro_f1(y, A.argmax(1)):.4f} per fold " + " ".join(f"{macro_f1(y[fold == k], A[fold == k].argmax(1)):.4f}" for k in range(5)))
    for rs in (False, True):
        G = ens(members, sbj, rs); lab = G.argmax(1)
        print(f"ensemble{' + resink' if rs else '':10s} F1 {macro_f1(y, lab):.4f} per fold " + " ".join(f"{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(5)))
else:
    out = args[0]; members = [load(a) for a in args[1:]]
    sbj = np.load(os.path.join(KEEP, "blend.npz"))["test_sbj"].astype(np.int64)
    G = ens(members, sbj, resink); lab = G.argmax(1)
    pd.DataFrame({"id": np.arange(len(lab)), "target_feature": lab.astype(int)}).to_csv(out, index=False); np.save(out.replace(".csv", "_Q.npy"), G.astype(np.float32))
    for nm, A, w in members:
        print(f"agreement with {nm}: {np.mean(lab == A.argmax(1)):.4f}")
    print("wrote", out, "null", np.mean(lab == 0).round(3))
