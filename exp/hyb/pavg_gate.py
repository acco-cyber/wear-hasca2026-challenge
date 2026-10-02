"""Ensemble of recipe outputs from independent base fits: average the pre-calibration P files written by graph_lab
(test mode), then the same count targets (ours-blended), Sinkhorn finish and top-2 gate.
python pavg_gate.py <out.csv> <ours.csv> <P1.npy> <P2.npy> [...] [--counts 0.3] [--gate 0.55:top2]"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_lab import make_targets, finish2, apply_gate, KEEP
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ours"); ap.add_argument("P", nargs="+")
ap.add_argument("--counts", type=float, default=0.3); ap.add_argument("--gate", default="0.55:top2")
a = ap.parse_args()
Ps = []
for p in a.P:
    x = np.load(p).astype(np.float64); Ps.append(x / x.sum(1, keepdims=True))
P = np.mean(Ps, 0)
sbj = np.load(os.path.join(KEEP, "blend.npz"))["test_sbj"].astype(np.int64)
ours = pd.read_csv(a.ours).sort_values("id").target_feature.to_numpy().astype(int); has = np.ones(len(ours), bool)
targets = make_targets(sbj, {}, ours, a.counts, {int(s): True for s in np.unique(sbj)})
base, Q = finish2(P, sbj, targets)
tau, rule = (a.gate.split(":") + ["plain"])[:2]
lab, g = apply_gate(base, Q, ours, has, tau, rule, sbj=sbj)
pd.DataFrame({"id": np.arange(len(lab)), "target_feature": lab.astype(int)}).to_csv(a.out, index=False)
np.save(a.out.replace(".csv", "_P.npy"), P.astype(np.float32))
print(f"wrote {a.out}; gated {g.mean():.4f}; null {np.mean(lab == 0):.3f}")
