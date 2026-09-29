"""Tiny single-activity 3rd session: detectable as a scene-distinct cluster?
(1) train analog sbj_2 (9/9/1, S3 = one re-done activity at another location): k-means on subject-centred PCA emb,
    print cluster runs along TRUE time next to label runs.
(2) test sbj 22 (S3 ~177 s, other location) / 23 (S3 ~107 s, other day, sunny): cluster sizes, compactness and the
    current best labels inside each cluster."""
import os
import numpy as np, pandas as pd
from common import *
import methods as M
sm, P = load_oof(); y, sbj, rec, st = sm["y"], sm["sbj"], sm["rec"], sm["start"]
emb = M.l2n(np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32))


def runs(a):
    cut = np.flatnonzero(np.diff(a) != 0) + 1; s = np.r_[0, cut]; e = np.r_[cut, len(a)]
    return [(int(a[i]), int(j - i)) for i, j in zip(s, e)]


ii = np.flatnonzero(rec == 14); ii = ii[np.argsort(st[ii])]
Z = M.pca(emb[ii] - emb[ii].mean(0), 20)
print("rec 14 (sbj 2) label runs >=10s:", [r for r in runs(y[ii]) if r[1] >= 10])
for k in (3, 4, 6, 8):
    lab = M.kmeans(Z, k, seed=0, restarts=3)
    # smooth for display: majority over 15 s
    sm_lab = np.array([np.bincount(lab[max(0, t - 7):t + 8], minlength=k).argmax() for t in range(len(lab))])
    print(f"k={k} sizes {np.bincount(lab).tolist()} smoothed runs>=20s:", [r for r in runs(sm_lab) if r[1] >= 20])
# test
bl = np.load(os.path.join(KEEP, "blend.npz")); tsb = bl["test_sbj"].astype(int)
te = M.l2n(np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32))
lab_best = pd.read_csv(r"E:\Claude code\wear\subs\sub_gl6_p03c03_top2_055.csv")["target_feature"].values
for s in (22, 23, 24, 25):
    jj = np.flatnonzero(tsb == s); Zt = M.pca(te[jj] - te[jj].mean(0), 20)
    for k in (4, 8, 12):
        lab = M.kmeans(Zt, k, seed=0, restarts=3)
        info = []
        for c in range(k):
            m = lab == c
            # compactness: mean distance to own centroid vs to nearest other centroid
            cls = np.bincount(lab_best[jj][m], minlength=19)
            top = np.argsort(-cls)[:3]
            info.append(f"{m.sum()}:{'/'.join(f'{t}x{cls[t]}' for t in top)}")
        print(f"test sbj {s} k={k}: " + "  ".join(info))
