"""Class-conditional gate on top of the confidence gate, with a split-half check (pair table learned on half of the
sessions, applied to the other half). python classgate_cv.py <their_oof_P.npy> <labels.pkl>"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import KEEP, N_CLS, TRAIN_SETS, calibrate, CFG, macro_f1
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb")
JOG = {1, 2, 3, 4, 5}
P = np.load(sys.argv[1]).astype(np.float64); P /= P.sum(1, keepdims=True)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
o2t = np.load(os.path.join(HYB, "rows.npz"))["ours_to_theirs"]
R = pickle.load(open(sys.argv[2], "rb")); ours = np.full(len(P), -1, np.int64); sess = np.full(len(P), -1, np.int64)
for i, (s, d) in enumerate(sorted(R.items())):
    rr = o2t[int(d["a"]):int(d["a"]) + int(d["n"])]; ours[rr] = np.asarray(d["lab"]); sess[rr] = i
S = ours >= 0; y = sm["y"]; sbj = sm["sbj"]
Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); Q = calibrate(Q / Q.sum(1, keepdims=True), sbj, TRAIN_SETS, CFG["per_ex"], CFG["null_min"])
base = Q.argmax(1); conf = Q.max(1); srt = np.argsort(-Q, 1); top2 = (srt[:, 0] == ours) | (srt[:, 1] == ours)
g0 = S & (conf < 0.6) & top2
def f1(lab, m=S):
    return macro_f1(y[m], lab[m])
def apply(g):
    lab = base.copy(); lab[g] = ours[g]; return lab
print(f"theirs {f1(base):.4f}  gate top2 0.6 {f1(apply(g0)):.4f}")
for cmax in (0.7, 0.8, 0.9, 1.01):
    g = g0 | (S & (base == 0) & np.isin(ours, list(JOG)) & (conf < cmax))
    print(f"+ null->jog rule conf<{cmax}: {f1(apply(g)):.4f} (extra gated {(g & ~g0).sum()})")
    g = g0 | (S & np.isin(ours, list(JOG)) & (conf < cmax))
    print(f"+ any->jog rule  conf<{cmax}: {f1(apply(g)):.4f} (extra gated {(g & ~g0).sum()})")

def learn_table(m, margin, min_n=80):
    dis = m & (base != ours); tab = set()
    for a_ in range(N_CLS):
        for b_ in range(N_CLS):
            k = dis & (base == a_) & (ours == b_)
            if k.sum() >= min_n and np.mean(ours[k] == y[k]) > np.mean(base[k] == y[k]) + margin:
                tab.add((a_, b_))
    return tab
def apply_table(tab, m, cmax):
    g = np.zeros(len(P), bool)
    for a_, b_ in tab:
        g |= m & (base == a_) & (ours == b_) & (conf < cmax)
    return g
half = sess % 2
for margin in (0.0, 0.1, 0.2):
    for cmax in (0.8, 0.9, 1.01):
        labs = base.copy(); labs[g0] = ours[g0]
        for h in (0, 1):
            tab = learn_table(S & (half != h), margin); g = apply_table(tab, S & (half == h), cmax) | (g0 & (half == h))
            labs[g] = ours[g]
        tab_all = learn_table(S, margin)
        print(f"pair table margin {margin} conf<{cmax}: split-half F1 {f1(labs):.4f}  | in-sample {f1(apply(g0 | apply_table(tab_all, S, cmax))):.4f}  pairs(all) {sorted(tab_all)}")
