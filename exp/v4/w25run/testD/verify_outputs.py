"""Sanity checks of the written link files: shapes, global indices within the subject, no self links, in-degree <= 1 per
matching, scores finite; plus label-free comparisons between variants and against the fits' own test links."""
import os, sys, json
import numpy as np
from tdlib import HERE, K7, K9, SENS

st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); tsbj = st["test_sbj"]; tsens = st["sensor_test"]; NT = len(tsbj)
sbj = st["oof_sbj"]; ts = st["true_succ"]; N = len(sbj)
own = {"K7": np.load(os.path.join(K7, "links.npz")), "K9": np.load(os.path.join(K9, "links.npz"))}
files = [f for f in sorted(os.listdir(HERE)) if f.startswith("links_test_") and f.endswith(".npz")]
first = {}
for f in files:
    z = np.load(os.path.join(HERE, f)); fit = "K9" if "K9" in f else "K7"
    for split, n_, sb in (("test", NT, tsbj), ("oof", N, sbj)):
        su, sc = z[f"{split}_succ"], z[f"{split}_score"]
        assert su.shape == (8, n_) and sc.shape == (8, n_), (f, split, su.shape)
        for k in range(8):
            m = su[k] >= 0
            assert (su[k][m] < n_).all() and (su[k][m] != np.flatnonzero(m)).all()
            assert (sb[su[k][m]] == sb[m]).all(), (f, split, k, "cross-subject link")
            assert np.bincount(su[k][m], minlength=n_).max() <= 1, (f, split, k, "in-degree > 1")
            assert np.isfinite(sc[k]).all()
    o = own[fit]["test_succ"][0]; s0 = z["test_succ"][0]; first[f] = s0
    h = ts >= 0; oo = z["oof_succ"][0]
    print(f"{f}: OK | test linked {np.mean(s0 >= 0):.4f}, changed vs own {np.mean(s0 != o):.4f}; by limb "
          + " ".join(f"{SENS[L]} {np.mean((s0 != o)[tsens == L]):.3f}" for L in range(4)) + f" | OOF exact {np.mean(oo[h] == ts[h]):.4f}")
for i, f in enumerate(files):
    for g in files[i + 1:]:
        if ("K9" in f) == ("K9" in g):
            print(f"agreement {f} vs {g}: {np.mean(first[f] == first[g]):.4f}")
