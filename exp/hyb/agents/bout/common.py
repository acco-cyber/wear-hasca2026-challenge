"""shared loaders for the bout agent"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import finish, KEEP, TRAIN_SETS, macro_f1, N_CLS, calibrate, CFG  # noqa

HB = r"E:\Claude code\wear\work\hanbat"
OUT = os.path.dirname(os.path.abspath(__file__))


def load_oof():
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    P = np.load(os.path.join(HB, "cv_all5_oof_P.npy")).astype(np.float64); P /= P.sum(1, keepdims=True)
    l0 = np.load(os.path.join(KEEP, "links_L0.npz"))
    d = dict(sm)
    d["P"] = P
    d["succ"] = l0["oof_succ"].astype(np.int64); d["score"] = l0["oof_score"].astype(np.float64)
    return d


def load_emb(split="oof"):
    return np.load(os.path.join(KEEP, f"{split}_emb.npy")).astype(np.float32)


def true_bouts(y, rec, st):
    """bout id per row (runs of one label in true time order within rec)"""
    bid = np.full(len(y), -1, np.int64); nb = 0
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r); ii = ii[np.argsort(st[ii], kind="stable")]
        yy = y[ii]; cut = np.flatnonzero(np.diff(yy) != 0) + 1
        seg = np.zeros(len(ii), np.int64); seg[cut] = 1; seg = np.cumsum(seg)
        bid[ii] = seg + nb; nb += seg[-1] + 1
    return bid


def cal_Q(P, sbj, sets):
    Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"])
    return calibrate(Q / Q.sum(1, keepdims=True), sbj, sets, CFG["per_ex"], CFG["null_min"])


def per_fold(y, pred, fold):
    return {int(f): round(macro_f1(y[fold == f], pred[fold == f]), 4) for f in np.unique(fold)}
