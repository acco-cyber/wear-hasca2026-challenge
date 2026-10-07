"""which predicted bout was recorded in a different scene?  residual = bout centroid - class prototype (both in
per-subject-centred video space); a bout from a separate session/date should have a residual unlike the subject's other
bouts.  Train check: sbj_2's unlabeled 3rd session (predicted class 2).  Test: sbj_22/23 session_3 candidates."""
from common import *

d = base(); y = d["oof_y"]; sb = d["oof_sbj"]; rec = d["oof_rec"]; ts = d["test_sbj"]
perm = np.load(os.path.join(OUT, "perm_oof2meta.npy"))
V = np.load(os.path.join(W, "data", "prep", "train_vid_mean768.npy")).astype(np.float32)[perm]
Vt = np.load(os.path.join(W, "data", "prep", "test_vid_mean768.npy")).astype(np.float32)
labo = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labo.npy")).astype(int)
labt = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labt.npy")).astype(int)


def centre(X, g):
    X = X.copy()
    for s in np.unique(g):
        X[g == s] -= X[g == s].mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


Vc = centre(V, rec); Vtc = centre(Vt, ts)
proto_all = {c: [Vc[(y == c) & (rec == r)].mean(0) for r in np.unique(rec) if ((y == c) & (rec == r)).sum() > 20] for c in range(1, 19)}
recs = np.unique(rec)


def outliers(Z, lab, g_excl=None):
    res = []
    for c in range(1, 19):
        m = lab == c
        if m.sum() < 20:
            continue
        pr = np.mean([p for r, p in zip([r for r in recs if ((y == c) & (rec == r)).sum() > 20], proto_all[c]) if r != g_excl], 0)
        res.append((c, Z[m].mean(0) - pr))
    R = np.array([v for _, v in res]); R /= np.linalg.norm(R, axis=1, keepdims=True); S = R @ R.T
    sc = [(round(float(1 - (S[i].sum() - 1) / (len(R) - 1)), 3), res[i][0]) for i in range(len(R))]
    return sorted(sc, reverse=True)


for r in recs:
    m = rec == r
    print(f"train rec {r} sbj {sb[m][0]}: top outlier bouts (score, class) {outliers(Vc[m], labo[m], r)[:4]}")
for s in (22, 23, 24, 25):
    m = ts == s
    print(f"TEST sbj {s}: top outlier bouts {outliers(Vtc[m], labt[m])[:5]}")
