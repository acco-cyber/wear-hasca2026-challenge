"""(a) can a subject's tiles be split into their sessions without labels?  validated on training recordings where the
session split is known from the block switch (clean AAAAAAAAA|BBBBBBBBB recordings, + the unlabeled 3rd sessions of
sbj_10 / sbj_2).  Features: per-tile VideoMAE mean (768), per-recording centred + whitened PCA."""
from common import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

d = base(); y = d["oof_y"]
perm = np.load(os.path.join(OUT, "perm_oof2meta.npy"))
V = np.load(os.path.join(W, "data", "prep", "train_vid_mean768.npy")).astype(np.float32)[perm]
E = np.load(os.path.join(K7, "oof_emb.npy")).astype(np.float32)
CLEAN = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]


def sessions_gt(r):
    o = ordered(d, r); b = BLK[y[o]]; nz = np.flatnonzero(b > 0)
    sw = [(nz[k] + nz[k + 1]) // 2 for k in range(len(nz) - 1) if b[nz[k]] != b[nz[k + 1]]]
    if r == 14:
        sw = [sw[-1]]                                    # sbj_2: the B tile at 246-450 sits inside session 1
    cuts = sw + ([2564] if r == 3 else []) + ([3445] if r == 14 else [])
    s = np.zeros(len(o), int)
    for c in cuts:
        s[c:] += 1
    return o, s


def feats(X, ncomp=48, power=0.5):
    X = X - X.mean(0); p = PCA(ncomp, random_state=0).fit(X); Z = p.transform(X) / (p.explained_variance_ ** power + 1e-8) ** 1
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def purity(s, c):
    tab = pd.crosstab(s, c).values
    from scipy.optimize import linear_sum_assignment
    r_, c_ = linear_sum_assignment(-tab)
    return tab[r_, c_].sum() / tab.sum()


rows = []
for r in (CLEAN if __name__ == "__main__" else []):
    o, s = sessions_gt(r); k = s.max() + 1
    res = {"rec": r, "sbj": int(d["oof_sbj"][o[0]]), "n": len(o), "sizes": list(np.bincount(s))}
    for nm, X in (("vid", V[o]), ("emb", E[o])):
        Z = feats(X)
        res[f"{nm}_km"] = purity(s, KMeans(k, n_init=10, random_state=0).fit_predict(Z))
        res[f"{nm}_spec"] = purity(s, SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="discretize").fit_predict(Z))
        nul = y[o] == 0
        lr = LogisticRegression(C=1.0, max_iter=2000)
        res[f"{nm}_probe_null"] = cross_val_score(lr, Z[nul], s[nul], cv=2).mean() if len(np.unique(s[nul])) > 1 else np.nan
        # train on non-null, test on null: does a session classifier generalise from activity tiles to null tiles?
        lr.fit(Z[~nul], s[~nul]); res[f"{nm}_act2null"] = (lr.predict(Z[nul]) == s[nul]).mean()
    rows.append(res); print(res, flush=True)
if __name__ == "__main__":
    df = pd.DataFrame(rows); print(df.drop(columns=["sizes"]).round(3).to_string())
    print(df.drop(columns=["sizes", "rec", "sbj", "n"]).mean().round(3))
    df.to_csv(os.path.join(OUT, "s03_cluster_train.csv"), index=False)
