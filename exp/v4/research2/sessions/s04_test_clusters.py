"""test subjects: look for visually distinct session clusters (sbj_22 / sbj_23 session_3 at another location / date)"""
from common import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA

d = base(); ts = d["test_sbj"]
V = np.load(os.path.join(W, "data", "prep", "test_vid_mean768.npy")).astype(np.float32)
lab = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labt.npy")).astype(int)
Qt = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_Qt.npy"))
print("predicted counts per subject (b4wa):")
for s in (22, 23, 24, 25):
    print(s, (ts == s).sum(), np.bincount(lab[ts == s], minlength=19).tolist())


def feats(X, ncomp=48, power=0.5):
    X = X - X.mean(0); p = PCA(ncomp, random_state=0).fit(X); Z = p.transform(X) / (p.explained_variance_ ** power + 1e-8)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


res = {}
for s in (22, 23, 24, 25):
    ii = np.flatnonzero(ts == s); Z = feats(V[ii])
    for k in (2, 3, 4, 5, 6):
        for nm, cl in (("km", KMeans(k, n_init=10, random_state=0).fit_predict(Z)),
                       ("spec", SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="discretize").fit_predict(Z))):
            sizes = np.bincount(cl)
            desc = []
            for c in np.argsort(sizes):
                jj = ii[cl == c]; cnt = np.bincount(lab[jj], minlength=19)
                top = [(int(a), int(cnt[a])) for a in np.argsort(cnt)[::-1][:4] if cnt[a] > 0]
                blk = np.bincount(BLK[lab[jj]], minlength=3)
                desc.append(f"{sizes[c]}:{top} blk0/A/B {blk.tolist()}")
            print(f"sbj {s} {nm} k={k}: " + " | ".join(desc), flush=True)
            res[f"{s}_{nm}_{k}"] = (ii, cl)
np.save(os.path.join(OUT, "s04_test_clusters.npy"), np.array([res], dtype=object), allow_pickle=True)
