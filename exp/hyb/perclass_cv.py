"""Per-class F1 on the 18 validated sessions: their graph labels, our decoder labels, and the gated combination; plus the
confusion pairs where the two families disagree. python perclass_cv.py <their_oof_P.npy> <labels.pkl>"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, KEEP, N_CLS, TRAIN_SETS, calibrate, CFG
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb")
CLS = ["null", "jog", "jog-arms", "jog-skip", "jog-side", "jog-butt", "str-tri", "str-lung", "str-shoul", "str-ham", "str-lumb",
       "pushup", "pushup-c", "situp", "situp-c", "burpee", "lunge", "lunge-c", "benchdip"]
P = np.load(sys.argv[1]).astype(np.float64); P /= P.sum(1, keepdims=True)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
o2t = np.load(os.path.join(HYB, "rows.npz"))["ours_to_theirs"]
R = pickle.load(open(sys.argv[2], "rb")); ours = np.full(len(P), -1, np.int64)
for s, d in R.items():
    ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
S = ours >= 0; y = sm["y"]; dd = dict(sbj=sm["sbj"], sets=TRAIN_SETS)
Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); Q = calibrate(Q / Q.sum(1, keepdims=True), dd["sbj"], dd["sets"], CFG["per_ex"], CFG["null_min"])
base = Q.argmax(1); conf = Q.max(1); srt = np.argsort(-Q, 1)
g = S & (conf < 0.6) & ((srt[:, 0] == ours) | (srt[:, 1] == ours)); gate = base.copy(); gate[g] = ours[g]

def pc(yt, yp):
    cm = np.bincount(yt * N_CLS + yp, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS); tp = np.diag(cm)
    return 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1), cm
ft, cmt = pc(y[S], base[S]); fo, cmo = pc(y[S], ours[S]); fg, _ = pc(y[S], gate[S])
print(f"{'class':10s} {'theirs':>7s} {'ours':>7s} {'gated':>7s}  n")
for c in range(N_CLS):
    print(f"{CLS[c]:10s} {ft[c]:7.3f} {fo[c]:7.3f} {fg[c]:7.3f}  {int((y[S] == c).sum())}")
print(f"macro     {ft.mean():7.4f} {fo.mean():7.4f} {fg.mean():7.4f}")
# where they disagree: who is right?
dis = S & (base != ours)
print(f"\ndisagreement rows {dis.sum()} ({dis.sum() / S.sum():.3f} of S): theirs right {np.mean(base[dis] == y[dis]):.3f}, ours right {np.mean(ours[dis] == y[dis]):.3f}, neither {np.mean((base[dis] != y[dis]) & (ours[dis] != y[dis])):.3f}")
for lo, hi in ((0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
    m = dis & (conf >= lo) & (conf < hi)
    if m.sum():
        print(f"  conf [{lo},{hi}): n={m.sum()} theirs right {np.mean(base[m] == y[m]):.3f} ours right {np.mean(ours[m] == y[m]):.3f}")
pairs = {}
for a_, b_ in zip(base[dis], ours[dis]):
    pairs[(a_, b_)] = pairs.get((a_, b_), 0) + 1
print("\ntop disagreement pairs (theirs -> ours): count, theirs-right, ours-right")
for (a_, b_), n in sorted(pairs.items(), key=lambda x: -x[1])[:14]:
    m = dis & (base == a_) & (ours == b_)
    print(f"  {CLS[a_]:10s} -> {CLS[b_]:10s} {n:5d}  {np.mean(base[m] == y[m]):.2f}  {np.mean(ours[m] == y[m]):.2f}")
