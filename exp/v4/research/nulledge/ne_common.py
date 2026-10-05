"""Shared helpers for the null-edge specialist: loading the cached baseline + both fits, k-step chain walks."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import numpy as np

K7 = r"E:\Claude code\wear\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = r"E:\Claude code\wear\work\v4\wear-v4-big-tf-opt-s9\keep4"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_all():
    base = dict(np.load(os.path.join(HERE, "base_cache.npz")))
    st7 = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); st9 = np.load(os.path.join(K9, "stage.npz"), allow_pickle=True)
    l7 = np.load(os.path.join(K7, "links.npz")); l9 = np.load(os.path.join(K9, "links.npz"))
    ts = np.load(os.path.join(K7, "tile_scalars.npz"))
    d = dict(base)
    for k in ["oof_y", "oof_sbj", "oof_fold", "test_sbj", "sensor_oof", "sensor_test", "ids", "true_succ", "oof_rec", "oof_start"]:
        d[k] = st7[k]
    for tag, st in (("7", st7), ("9", st9)):
        for k in ["B2_OOF", "B2_TEST", "oof_logp", "test_logp", "TA_OOF", "TA_TEST", "tab_S3_oof", "tab_S3_test", "QB_OOF", "QB_TEST"]:
            d[k + "_" + tag] = st[k].astype(np.float32)
    d["Lo"] = np.concatenate([l7["oof_succ"], l9["oof_succ"]]).astype(np.int64)
    d["Lt"] = np.concatenate([l7["test_succ"], l9["test_succ"]]).astype(np.int64)
    d["Lso"] = np.concatenate([l7["oof_score"], l9["oof_score"]]).astype(np.float32)
    d["Lst"] = np.concatenate([l7["test_score"], l9["test_score"]]).astype(np.float32)
    for k in ["ener", "post", "vmot", "vmean"]:
        d["oof_" + k] = ts["oof_" + k]; d["test_" + k] = ts["test_" + k]
    return d


def prev_of(succ):
    n = len(succ); p = np.full(n, -1, np.int64); m = succ >= 0; p[succ[m]] = np.flatnonzero(m); return p


def walks(succ, H):
    """k-step successors / predecessors for k = 1..H (index -1 once the chain ends): arrays (H, n)"""
    prv = prev_of(succ)
    F = np.full((H, len(succ)), -1, np.int64); B = np.full((H, len(succ)), -1, np.int64)
    cf, cb = succ.copy(), prv.copy()
    for k in range(H):
        F[k] = cf; B[k] = cb
        cf = np.where(cf >= 0, succ[np.maximum(cf, 0)], -1); cb = np.where(cb >= 0, prv[np.maximum(cb, 0)], -1)
    return F, B
