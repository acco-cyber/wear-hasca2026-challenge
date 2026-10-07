"""Shared loaders for the w25 feasibility simulation (training sessions, all four limbs, 1-s tiles on the OOF grid)."""
import os, glob
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"
OUT = os.path.join(W, "exp", "v4", "research2", "w25feas")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
# CSV column blocks in file order = the pipeline's sensor index: 0 right_arm, 1 right_leg, 2 left_leg, 3 left_arm
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]
COLS = [f"{l}_acc_{a}" for l in SENS for a in "xyz"]
STEMS = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(W, "data", "train", "inertial_feat", "sbj_*.csv")))
CLASSES = None


def load_stage(d=K7):
    st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True)
    keep = ["oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "true_succ", "sensor_oof"]
    return {k: st[k].astype(np.int64) for k in keep}


def load_tiles(rec, n_expected=None):
    """(4, n, 50, 3) float32 tiles of one recording on the 50-sample grid, NaN -> linear fill / 0"""
    A = pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{STEMS[rec]}.csv"), usecols=COLS)[COLS].to_numpy(np.float32)
    n = len(A) // 50 if n_expected is None else n_expected
    A = A[: n * 50]
    if np.isnan(A).any():
        A = pd.DataFrame(A).interpolate(limit_direction="both").fillna(0.0).to_numpy(np.float32)
    return np.transpose(A.reshape(n, 50, 4, 3), (2, 0, 1, 3)).copy()


def subject_tiles(S, s):
    """all OOF rows of subject s in (rec, start) order + their 4-limb tiles (4, n, 50, 3)"""
    ii = np.flatnonzero(S["oof_sbj"] == s)
    ii = ii[np.lexsort((S["oof_start"][ii], S["oof_rec"][ii]))]
    parts = []
    for r in np.unique(S["oof_rec"][ii]):
        jj = ii[S["oof_rec"][ii] == r]
        t = load_tiles(r, int(S["oof_start"][jj].max() // 50 + 1))
        parts.append(t[:, S["oof_start"][jj] // 50])
    return ii, np.concatenate(parts, 1)
