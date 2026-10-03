"""Are our chain decoder's per-(subject, exercise) window counts a useful, independent estimate of bout length?
Compare count errors (MAE vs true) of: fixed 97, the public 0.927 notebook's final OOF labels, our mrf4 decoder labels
(18 sessions), and averages; then re-calibrate their OOF probabilities with blended count targets and score."""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import macro_f1, KEEP, TRAIN_SETS, N_CLS, CFG
from graph_lab import calibrate_targets
W = r"E:\Claude code\wear"
O = np.load(os.path.join(W, "public_src", "good927", "out", "final_probabilities.npz"))["oof"].astype(np.float64)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}; y, sbj, fold = sm["y"], sm["sbj"], sm["fold"]
o2t = np.load(os.path.join(W, "exp", "hyb", "rows.npz"))["ours_to_theirs"]; R = pickle.load(open(os.path.join(W, "exp", "pl", "labels_v3bv1f.pkl"), "rb"))
ours = np.full(len(y), -1, np.int64)
for s, d in R.items():
    ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
base = O.argmax(1); subs = [int(s) for s in np.unique(sbj) if (ours[sbj == s] >= 0).all()]
print("subjects fully covered by our decoder labels:", subs)
rows = []
for s in subs:
    m = sbj == s; ns = TRAIN_SETS.get(s, 1)
    for c in range(1, N_CLS):
        rows.append((s, c, (y[m] == c).sum() / ns, (base[m] == c).sum() / ns, (ours[m] == c).sum() / ns))
A = np.array(rows, float); true, th, ou = A[:, 2], A[:, 3], A[:, 4]
ok = (true > 55) & (true < 150)
print(f"pairs {len(A)} (regular {ok.sum()}): MAE fixed 97 {np.abs(97 - true[ok]).mean():.2f} | theirs {np.abs(th - true)[ok].mean():.2f} | ours {np.abs(ou - true)[ok].mean():.2f} "
      f"| mean(theirs, ours) {np.abs((th + ou) / 2 - true)[ok].mean():.2f} | corr of errors {np.corrcoef((th - true)[ok], (ou - true)[ok])[0, 1]:.3f}")
for clipmax in (140, 160):
    oc = np.clip(ou, 60, clipmax)
    for w in (0.25, 0.5):
        print(f"  blend (1-{w})*theirs + {w}*clip(ours, 60, {clipmax}): MAE {np.abs((1 - w) * th + w * oc - true)[ok].mean():.2f}")
# re-calibrate their probabilities with blended targets (only for the covered subjects) and score
M = np.isin(sbj, subs)
def targets_from(cnt_by_key):
    T = {}
    for s in np.unique(sbj):
        m = sbj == s; n = int(m.sum()); ns = TRAIN_SETS.get(int(s), 1)
        per = np.array([cnt_by_key.get((int(s), c), (base[m] == c).sum() / ns) for c in range(1, N_CLS)]) * ns
        null = max(n - per.sum(), CFG["null_min"] * n); T[int(s)] = np.r_[null, per * (n - null) / per.sum()]
    return T
key = [(int(a), int(b)) for a, b in A[:, :2]]
print(f"\ntheir OOF F1 on covered subjects: {macro_f1(y[M], base[M]):.4f}")
for nm, cnt in (("their own counts (identity check)", th), ("true counts (oracle)", true), ("ours", np.clip(ou, 60, 150)),
                ("0.75 theirs + 0.25 ours", 0.75 * th + 0.25 * np.clip(ou, 60, 150)), ("0.5 theirs + 0.5 ours", 0.5 * th + 0.5 * np.clip(ou, 60, 150))):
    T = targets_from(dict(zip(key, cnt))); Q = calibrate_targets(O.copy(), sbj, T); lab = Q.argmax(1)
    print(f"  re-calibrated with {nm}: F1 {macro_f1(y[M], lab[M]):.4f}")
