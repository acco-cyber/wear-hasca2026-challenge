"""Average several pre-calibration graph probability files (kernel P_test + local re-runs), then sharpen + Sinkhorn
(finish) -> submission.  python pavg.py out.csv file1.npy[:w] file2.npy[:w] ..."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, KEEP, N_CLS
out = sys.argv[1]; Ps, ws = [], []
for spec in sys.argv[2:]:
    p, w = (spec.rsplit(":", 1) + ["1.0"])[:2] if ":" in spec[2:] else (spec, "1.0")
    P = np.load(p).astype(np.float64); P /= P.sum(1, keepdims=True); Ps.append(P); ws.append(float(w))
ws = np.array(ws) / np.sum(ws); P = sum(w * p for w, p in zip(ws, Ps))
bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64)
lab = finish(P, dict(sbj=sbj, sets={}))
ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True)
pd.DataFrame({"id": ids, "target_feature": lab.astype(int)}).to_csv(out, index=False)
ref = bl["labels"]; print("wrote", out, "agreement with kernel labels", np.mean(lab == ref).round(4), "null", np.mean(lab == 0).round(3))
