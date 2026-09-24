"""Build a (12234,19) probability file: log-blend of model test.npy files + optional hard-vote bonuses.
python make_probs.py out.npy --tags lgbm_v1:0.8,fusion_v1:0.2 [--extra path.npy:w ...] --votes csv:w,csv:w"""
import os, argparse, numpy as np, pandas as pd
W = r"E:\Claude code\wear\work"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--tags", default="lgbm_v1:0.8,fusion_v1:0.2")
ap.add_argument("--extra", default=None, help="comma list of npy_path:weight (log-blended)"); ap.add_argument("--votes", default=None)
a = ap.parse_args()
L = 0
for spec in a.tags.split(","):
    t, w = spec.rsplit(":", 1); L = L + float(w) * np.log(np.clip(np.load(os.path.join(W, t, "test.npy")), 1e-6, 1))
if a.extra:
    for spec in a.extra.split(","):
        p, w = spec.rsplit(":", 1); L = L + float(w) * np.log(np.clip(np.load(p), 1e-6, 1))
if a.votes:
    for spec in a.votes.split(","):
        p, w = spec.rsplit(":", 1)
        v = pd.read_csv(p).sort_values("id").iloc[:, 1].to_numpy().astype(int); L[np.arange(len(v)), v] += float(w)
P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
np.save(a.out, P.astype(np.float32)); print("saved", a.out, P.shape, "argmax null", round(float((P.argmax(1) == 0).mean()), 3))
