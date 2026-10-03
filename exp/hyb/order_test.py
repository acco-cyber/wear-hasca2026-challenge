"""Test: our mrf4 chain decoder on a graph P along a given test link set, per subject -> submission CSV.
python order_test.py <P_test.npy> <links.npz> <out.csv> [--variant mrf4] [--lam x] [--ns x] [--lo n] [--hi n]"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from order_decode import decode_links
from hanbat_stack import KEEP
ap = argparse.ArgumentParser(); ap.add_argument("P"); ap.add_argument("links"); ap.add_argument("out"); ap.add_argument("--variant", default="mrf4")
ap.add_argument("--lam", type=float, default=None); ap.add_argument("--ns", type=float, default=None); ap.add_argument("--lo", type=int, default=None); ap.add_argument("--hi", type=int, default=None)
a = ap.parse_args()
P = np.load(a.P).astype(np.float64); P /= P.sum(1, keepdims=True)
z = np.load(a.links); succ, score = z["succ"].astype(np.int64), z["score_qn"].astype(np.float32)
bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64)
over = {k: v for k, v in (("lam", a.lam), ("ns", a.ns), ("lo", a.lo), ("hi", a.hi)) if v is not None}
lab = decode_links(P, succ, score, sbj, sbj, a.variant, None, None, over)
ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True); assert (ids == np.arange(len(ids))).all()
pd.DataFrame({"id": ids, "target_feature": lab.astype(int)}).to_csv(a.out, index=False)
ref = bl["labels"]
print(f"wrote {a.out}; linked {np.mean(succ >= 0):.3f}; agreement with kernel labels {np.mean(lab == ref):.4f}; null {np.mean(lab == 0):.3f}; per subject null "
      + str({int(s): round(float(np.mean(lab[sbj == s] == 0)), 3) for s in np.unique(sbj)}))
