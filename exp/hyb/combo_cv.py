"""CV of combining the Hanbat graph probabilities (their OOF rows) with OUR decoder's OOF labels (18 sim sessions):
 (c) product P * (onehot(ours)*(1-eps)+eps/19)^w before finish(),  (d) confidence gate after finish().
python combo_cv.py <their_oof_P.npy> <labels_pkl>"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, macro_f1, KEEP, N_CLS, TRAIN_SETS, calibrate, _norm_rows, CFG
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb")
P = np.load(sys.argv[1]).astype(np.float64); P /= P.sum(1, keepdims=True)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
rows = np.load(os.path.join(HYB, "rows.npz")); o2t = rows["ours_to_theirs"]
R = pickle.load(open(sys.argv[2], "rb"))
ours = np.full(len(P), -1, np.int64)
for s, d in R.items():
    a, n = int(d["a"]), int(d["n"]); ours[o2t[a:a + n]] = np.asarray(d["lab"])
S = ours >= 0
y = sm["y"]; dd = dict(sbj=sm["sbj"], sets=TRAIN_SETS)
print(f"rows with our labels: {S.sum()} of {len(P)}; sessions {len(R)}")
base = finish(P, dd)
print(f"their graph alone: F1(all) {macro_f1(y, base):.4f}  F1(S) {macro_f1(y[S], base[S]):.4f}")
print(f"ours alone on S:   F1(S) {macro_f1(y[S], ours[S]):.4f}   agreement ours vs theirs on S {np.mean(ours[S] == base[S]):.4f}")
eps = 0.1
oh = np.full((len(P), N_CLS), eps / N_CLS); oh[S, ours[S]] += 1 - eps
for w in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    Pc = P.copy(); Pc[S] = P[S] * oh[S] ** w; Pc = _norm_rows(Pc)
    lab = finish(Pc, dd)
    print(f"product w={w}: F1(S) {macro_f1(y[S], lab[S]):.4f}")
# gate on calibrated confidence
Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); Q = calibrate(Q / Q.sum(1, keepdims=True), dd["sbj"], dd["sets"], CFG["per_ex"], CFG["null_min"])
conf = Q.max(1)
rank_ours = np.full(len(P), 99)
srt = np.argsort(-Q, 1)
for k in range(5):
    rank_ours[S & (srt[:, k] == ours)] = np.minimum(rank_ours[S & (srt[:, k] == ours)], k)
q_ours = np.where(S, Q[np.arange(len(P)), np.maximum(ours, 0)], 0.0)
for tau in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
    res = []
    for nm, extra in (("plain", np.ones(len(P), bool)), ("top2", rank_ours <= 1), ("top3", rank_ours <= 2),
                      ("q>=0.05", q_ours >= 0.05), ("q>=0.1", q_ours >= 0.1)):
        lab = base.copy(); g = S & (conf < tau) & extra; lab[g] = ours[g]
        res.append(f"{nm} {macro_f1(y[S], lab[S]):.4f} ({g.sum() / S.sum():.3f})")
    print(f"gate tau={tau}: " + " | ".join(res))
