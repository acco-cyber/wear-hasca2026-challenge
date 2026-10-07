"""Sanity check on the real 2nd-challenge test data (read-only): per (subject, limb) naive within-limb continuity
Hungarian over ALL 2025 tiles, then compare with the s7 fit's first L3 test matching for 2026 links whose two ends are
the same limb and both have exact 2025 twins. Also reports the OOF exact rate of same-limb assigned links (s7 fit) so
agreement can be read as P(link exact) * P(chain exact)."""
import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_limbchain import cost_matrix

W = r"E:\Claude code\wear"; K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
w = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25, s25, l25 = w["acc"], w["sbj"], w["limb"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); tw = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float32)
exact = np.abs(A26 - A25[tw]).max((1, 2)) < 1e-4
st = np.load(os.path.join(K7, "stage.npz")); L = np.load(os.path.join(K7, "links.npz"))
sen_t, sen_o = st["sensor_test"], st["sensor_oof"]
# OOF: same-limb share and exact rate of assigned links (matching 0)
os_, ts = L["oof_succ"][0], st["true_succ"]
has = (os_ >= 0) & (ts >= 0)
same_o = has & (sen_o == sen_o[np.clip(os_, 0, None)])
print(f"OOF assigned links: exact {np.mean(os_[has] == ts[has]):.4f} | same-limb share {same_o.sum() / has.sum():.3f} | exact among same-limb {np.mean(os_[same_o] == ts[same_o]):.4f}")
# within-limb chains on 2025
succ25 = np.full(len(A25), -1)
for s in np.unique(s25):
    for l in range(4):
        ii = np.flatnonzero((s25 == s) & (l25 == l))
        C = cost_matrix(A25[ii], "lin"); r, c = linear_sum_assignment(C)
        succ25[ii[r]] = ii[c]
inv = {int(t): i for i, t in enumerate(tw) if exact[i]}
ts_ = L["test_succ"][0]
ok = (ts_ >= 0)
a = np.flatnonzero(ok & exact)
b = ts_[a]
same = (sen_t[a] == sen_t[b]) & exact[b]
a, b = a[same], b[same]
agree = succ25[tw[a]] == tw[b]
print(f"TEST same-limb assigned links with exact twins: n {len(a)} | 2025-chain agrees {agree.mean():.4f}")
# how often does the 2025 chain successor of a 2026 tile land on another 2026 tile (25% expected if limbs random)
nxt = succ25[tw[exact]]
print("2025-chain successor is itself a 2026 tile:", np.mean([int(t) in inv for t in nxt]).round(4))
