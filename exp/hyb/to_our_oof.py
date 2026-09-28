"""Convert a (69326,19) OOF probability/log-prob file in the Hanbat row order into our (69326,4,19) per-limb OOF layout
(our train_meta rows, the same probabilities in all four limb slots), or build our own reference OOF blend, so that
exp/pl/sim_final.py can score our decoder on any base.
python to_our_oof.py theirs <file.npy> <out.npy> [--log]      (their order -> ours)
python to_our_oof.py ours <out.npy>                            (0.5 v3b + 0.3 v1 + 0.2 fusion log-blend, per limb)"""
import os, sys
import numpy as np
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb")

if sys.argv[1] == "ours":
    parts = [(os.path.join(W, "exp", "base", "v3b", "oof.npy"), 0.5), (os.path.join(W, "work", "lgbm_v1", "oof.npy"), 0.3),
             (os.path.join(W, "work", "fusion_v1", "oof.npy"), 0.2)]
    L = 0
    for p, w in parts:
        L = L + w * np.log(np.clip(np.load(p).astype(np.float64), 1e-6, 1))
    P = np.exp(L - np.nanmax(L, 2, keepdims=True)); P /= np.nansum(P, 2, keepdims=True)
    np.save(sys.argv[2], P.astype(np.float32)); print("saved", sys.argv[2], P.shape)
else:
    rows = np.load(os.path.join(HYB, "rows.npz")); t2o = rows["their_to_ours"]
    X = np.load(sys.argv[2]).astype(np.float64)
    if "--log" in sys.argv:
        X = np.exp(X - X.max(1, keepdims=True))
    X = X / X.sum(1, keepdims=True)
    out = np.full((len(X), 4, 19), np.nan, np.float32)
    out[t2o] = X[:, None, :].astype(np.float32)
    assert np.isfinite(out).all()
    np.save(sys.argv[3], out); print("saved", sys.argv[3], out.shape)
