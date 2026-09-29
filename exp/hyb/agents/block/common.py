import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP, finish, TRAIN_SETS, macro_f1, N_CLS  # noqa
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]
BLK = np.zeros(19, int); BLK[B1] = 1; BLK[B2] = 2
M1 = np.zeros(19, bool); M1[B1] = True; M2 = np.zeros(19, bool); M2[B2] = True


def load_oof():
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy").astype(np.float64); P /= P.sum(1, keepdims=True)
    return sm, P


def true_half(sm):
    """oracle block for EVERY window (incl. null): block of the nearest activity window in true time order"""
    y, rec, st = sm["y"], sm["rec"], sm["start"]
    out = np.zeros(len(y), int)
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r); ii = ii[np.argsort(st[ii])]
        a = np.flatnonzero(y[ii] > 0); t = np.arange(len(ii))
        pos = np.searchsorted(a, t); lo = a[np.clip(pos - 1, 0, len(a) - 1)]; hi = a[np.clip(pos, 0, len(a) - 1)]
        near = np.where(np.abs(t - lo) <= np.abs(hi - t), lo, hi)
        out[ii] = BLK[y[ii][near]]
    return out


def apply_prior(P, p1, a, null_mode="keepnull", mask=None):
    """P[:,B1] *= p1^a, P[:,B2] *= (1-p1)^a; renormalise.  null_mode:
      'raw'      = literal spec form (null untouched in the unnormalised product -> null gains when p1 ~ 0.5)
      'keepnull' = rescale the activity part so every row keeps its original P(null) (pure B1<->B2 redistribution)
    mask: rows to touch (label-free); a may be a per-row array."""
    p1 = np.clip(p1, 1e-4, 1 - 1e-4)
    if mask is not None:
        p1 = np.where(mask, p1, 0.5)
    p1 = p1[:, None]; a = np.asarray(a, float)
    a = a[:, None] if a.ndim == 1 else a
    Q = P.copy()
    Q[:, M1] *= p1 ** a; Q[:, M2] *= (1 - p1) ** a
    if null_mode == "keepnull":
        act0 = P[:, 1:].sum(1, keepdims=True); act1 = Q[:, 1:].sum(1, keepdims=True)
        Q[:, 1:] *= act0 / np.maximum(act1, 1e-300)
        if mask is not None:
            Q[~mask] = P[~mask]
    return Q / Q.sum(1, keepdims=True)


def score(y, fold, sbj, P, sets=TRAIN_SETS):
    pred = finish(P, dict(sbj=sbj, sets=sets))
    return macro_f1(y, pred), [macro_f1(y[fold == f], pred[fold == f]) for f in range(5)], pred
