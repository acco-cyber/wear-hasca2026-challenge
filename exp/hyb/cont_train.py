"""How well does same-sensor sample continuity identify the next second? Train sessions (true order known), per limb:
cost(i->j) = |a_i[last] - a_j[0]|^2 + |extrapolated a_i -> a_j[0]|^2 (+ magnitude smoothness). Rank of the true
successor, and exact accuracy of a per-limb Hungarian assignment.  python cont_train.py sbj_5 sbj_9 ..."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
W = r"E:\Claude code\wear"
LIMB_COLS = {"left_arm": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"], "left_leg": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
             "right_arm": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"], "right_leg": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"]}


def cost_matrix(T, H, H2, T2):
    """T: tails (n,3) last sample; T2: second-to-last; H: heads (n,3) first sample; H2: second sample"""
    ext = 2 * T - T2                                # linear extrapolation of the tail to the next sample
    back = 2 * H - H2                                # backward extrapolation of the head to the previous sample
    d1 = ((T[:, None, :] - H[None, :, :]) ** 2).sum(2)
    d2 = ((ext[:, None, :] - H[None, :, :]) ** 2).sum(2)
    d3 = ((T[:, None, :] - back[None, :, :]) ** 2).sum(2)
    return d1 + 0.5 * d2 + 0.5 * d3


for sess in sys.argv[1:]:
    df = pd.read_csv(os.path.join(W, "data", "train", "inertial_feat", f"{sess}.csv"))
    n = len(df) // 50
    for limb, cols in LIMB_COLS.items():
        a = df[cols].to_numpy(np.float32)[: n * 50].reshape(n, 50, 3)
        ok = np.isfinite(a).all((1, 2)); idx = np.flatnonzero(ok); a = a[ok]; m = len(a)
        C = cost_matrix(a[:, -1], a[:, 0], a[:, 1], a[:, -2]); np.fill_diagonal(C, np.inf)
        true_next = np.where(np.r_[np.diff(idx) == 1, False], np.arange(1, m + 1), -1)   # local index of true successor
        has = true_next >= 0
        rank = (C[has] < C[has, true_next[has]][:, None]).sum(1)
        r, c = linear_sum_assignment(np.where(np.isfinite(C), C, 1e6))
        succ = np.full(m, -1); succ[r] = c
        exact = np.mean(succ[has] == true_next[has])
        mag = np.linalg.norm(a, axis=2); dyn = mag.std(1) > 0.08
        print(f"{sess} {limb:9s}: n={m} true-succ rank1 {np.mean(rank == 0):.3f} top3 {np.mean(rank < 3):.3f} | Hungarian exact {exact:.3f} "
              f"| dynamic windows ({dyn.mean():.2f}): exact {np.mean(succ[has & dyn] == true_next[has & dyn]):.3f}; static: {np.mean(succ[has & ~dyn] == true_next[has & ~dyn]):.3f}")
