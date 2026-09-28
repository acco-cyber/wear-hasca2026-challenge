"""Confidence gate on the test set: their graph probabilities P (pre-calibration) -> sharpen + Sinkhorn -> labels and
confidence; where confidence < tau, take OUR decoder's label from a submission CSV.
python gate_test.py <P.npy> <ours.csv> <tau> <out.csv>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, KEEP, CFG, calibrate, N_CLS
P = np.load(sys.argv[1]).astype(np.float64); P /= P.sum(1, keepdims=True)
ours = pd.read_csv(sys.argv[2]).sort_values("id").target_feature.to_numpy().astype(int)
tau = float(sys.argv[3]); out = sys.argv[4]; rule = sys.argv[5] if len(sys.argv) > 5 else "plain"
bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64); dd = dict(sbj=sbj, sets={})
Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); Q = calibrate(Q / Q.sum(1, keepdims=True), sbj, {}, CFG["per_ex"], CFG["null_min"])
base = Q.argmax(1); conf = Q.max(1)
g = conf < tau
if rule == "top2":                       # our label must be their 1st or 2nd choice
    srt = np.argsort(-Q, 1); g &= (srt[:, 0] == ours) | (srt[:, 1] == ours)
elif rule.startswith("q"):               # our label must have calibrated probability >= q
    g &= Q[np.arange(len(Q)), ours] >= float(rule[1:])
lab = base.copy(); lab[g] = ours[g]
ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True); assert (ids == np.arange(len(ids))).all()
pd.DataFrame({"id": ids, "target_feature": lab.astype(int)}).to_csv(out, index=False)
print(f"tau={tau}: gated {g.mean():.4f} (changed {np.mean(lab != base):.4f}); base agrees with kernel labels {np.mean(base == bl['labels']):.4f}; "
      f"conf quantiles 10/25/50 = {np.quantile(conf, [0.1, 0.25, 0.5]).round(3).tolist()}; null {np.mean(lab == 0):.3f}; per-subject gated "
      + str({int(s): round(float(g[sbj == s].mean()), 3) for s in np.unique(sbj)}))
