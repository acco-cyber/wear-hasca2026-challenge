#!/usr/bin/env python3
"""Validate LAP-based temporal order reconstruction on TRAIN data (known order).
Take contiguous chunks of a subject's limb windows, shuffle, LAP-reconstruct, measure accuracy."""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

rng = np.random.RandomState(42)

def load_subject_limb(sbj, limb):
    df = pd.read_csv(f"/home/z/my-project/data/train/inertial_feat/{sbj}.csv")
    cols = [f"{limb}_acc_{a}" for a in "xyz"]
    V = df[cols].to_numpy(dtype=np.float32)
    return V.reshape(-1, 25, 3)  # (n_win, 25, 3) true order

def lap_reconstruct(chunk):
    """chunk: (n, 25, 3). Cost[i,j] = boundary cost of j following i."""
    n = len(chunk)
    head = chunk[:, :2, :].reshape(n, -1)   # first 2 samples
    tail = chunk[:, -2:, :].reshape(n, -1)  # last 2 samples
    # cost[i,j] = ||head_j - tail_i|| ; forbid self-loop with +inf diag
    D = np.linalg.norm(head[:, None, :] - tail[None, :, :], axis=2)
    np.fill_diagonal(D, 1e6)
    r, c = linear_sum_assignment(D)
    return r, c, D

def eval_chunk(V, start, length):
    chunk = V[start:start + length]
    r, c, D = lap_reconstruct(chunk)
    # true successor pairs: i -> i+1 (within chunk)
    true_pairs = set(zip(range(length - 1), range(1, length)))
    pred_pairs = set(zip(r.tolist(), c.tolist()))
    inter = true_pairs & pred_pairs
    # also: fraction of predicted edges that are "true-adjacent" (|i-j|==1)
    adj = sum(1 for i, j in pred_pairs if abs(i - j) == 1)
    return len(inter) / (length - 1), adj / length

if __name__ == "__main__":
    for sbj, limb in [("sbj_0", "right_leg"), ("sbj_5", "left_arm")]:
        V = load_subject_limb(sbj, limb)
        for L in [300, 800]:
            accs, adjs = [], []
            for start in [0, 1500, 3000]:
                if start + L > len(V): continue
                a, b = eval_chunk(V, start, L)
                accs.append(a); adjs.append(b)
            print(f"{sbj} {limb} chunk={L}: true-succ-recovered={np.mean(accs):.3f} adj-rate={np.mean(adjs):.3f}")
