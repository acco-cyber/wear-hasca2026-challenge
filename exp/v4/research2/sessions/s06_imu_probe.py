"""does the inertial gravity direction (sensor re-mounting between sessions) separate sessions?  supervised
time-blocked probe on training null tiles, per limb, + unsupervised 2-means on the same features"""
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from s03_cluster_train import sessions_gt, purity, CLEAN

d = base(); y = d["oof_y"]
perm = np.load(os.path.join(OUT, "perm_oof2meta.npy"))
X = np.load(os.path.join(W, "data", "prep", "train_imu.npy"))[perm].astype(np.float32)      # (n, 4, 50, 3)
mu = np.nanmean(X, 2); sd = np.nanstd(X, 2)
g = mu / (np.linalg.norm(mu, axis=2, keepdims=True) + 1e-6)
F = np.concatenate([mu, g, sd], 2)                       # (n, 4, 9)
rows = []
for r in CLEAN:
    o, s = sessions_gt(r); nul = (y[o] == 0)
    calm = nul & (sd[o].mean((1, 2)) < 0.15)            # quasi-static null tiles (standing, resting)
    res = {"rec": r, "sbj": int(d["oof_sbj"][o[0]]), "calm": int(calm.sum())}
    for limb in range(4):
        Z = F[o, limb]; ok = ~np.isnan(Z).any(1)
        m = calm & ok
        if len(np.unique(s[m])) < 2:
            continue
        # time-blocked 2-fold: alternate 120-s chunks
        fold = (np.arange(len(o)) // 120) % 2; acc = []
        for k in (0, 1):
            tr, te = m & (fold != k), m & (fold == k)
            if len(np.unique(s[tr])) < 2 or te.sum() == 0:
                continue
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], s[tr]); acc.append((lr.predict(Z[te]) == s[te]).mean())
        res[f"probe_l{limb}"] = float(np.mean(acc))
        res[f"km_l{limb}"] = purity(s[m], KMeans(s.max() + 1, n_init=10, random_state=0).fit_predict(Z[m][:, 3:6]))
    rows.append(res); print(res, flush=True)
df = pd.DataFrame(rows); print(df.round(3).to_string()); print(df.mean().round(3))
df.to_csv(os.path.join(OUT, "s06_imu_probe.csv"), index=False)
