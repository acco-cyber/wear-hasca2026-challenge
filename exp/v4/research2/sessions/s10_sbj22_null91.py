"""sbj_22: what is the visually isolated 91-tile all-null cluster?  candidate for the separate-location session_3
(or the 180 unlisted seconds).  Links into / out of it, its B2 second choices, and which predicted bouts look like it."""
from common import *
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

d = base(); ts = d["test_sbj"]
Vt = np.load(os.path.join(W, "data", "prep", "test_vid_mean768.npy")).astype(np.float32)
lab = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labt.npy")).astype(int)
Qt = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_Qt.npy"))
st = np.load(os.path.join(K7, "stage.npz")); Bt = np.exp(st["B2_TEST"].astype(np.float64))
lk = np.load(os.path.join(K7, "links.npz")); su = lk["test_succ"]
sens = d["sensor_test"]


def feats(X, ncomp=48, power=0.5):
    X = X - X.mean(0); p = PCA(ncomp, random_state=0).fit(X); Z = p.transform(X) / (p.explained_variance_ ** power + 1e-8)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


for s in (22, 23):
    ii = np.flatnonzero(ts == s); Z = feats(Vt[ii])
    cl = SpectralClustering(3, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="discretize").fit_predict(Z)
    sizes = np.bincount(cl); c0 = np.argmin(sizes); jj = ii[cl == c0]
    print(f"sbj {s}: smallest spectral cluster n={len(jj)} labels {np.bincount(lab[jj], minlength=19).tolist()}")
    print("   mean B2 prob (top 5):", sorted(((round(float(v), 3), c) for c, v in enumerate(Bt[jj].mean(0))), reverse=True)[:5])
    print("   mean Q prob (top 5):", sorted(((round(float(v), 3), c) for c, v in enumerate(Qt[jj].mean(0))), reverse=True)[:5])
    print("   sensors:", np.bincount(sens[jj], minlength=4).tolist())
    mem = np.zeros(len(ts), bool); mem[jj] = True
    out_ = su[:, jj].ravel(); out_ = out_[out_ >= 0]; leave = out_[~mem[out_]]
    inn = np.flatnonzero(np.isin(su, jj).any(0) & ~mem)
    print(f"   links leaving: {len(leave)} of {len(out_)} -> labels {np.bincount(lab[leave], minlength=19).tolist()}")
    print(f"   tiles linking INTO it from outside: {len(inn)} labels {np.bincount(lab[inn], minlength=19).tolist()}")
    # bout similarity: centroid cosine of every predicted class (in the subject-centred whitened space) to the cluster
    cen = Z[cl == c0].mean(0)
    sims = []
    for c in range(19):
        m = (lab[ii] == c) & (cl != c0)
        if m.sum() > 10:
            v = Z[m].mean(0); sims.append((round(float(v @ cen / np.linalg.norm(v) / np.linalg.norm(cen)), 3), c, int(m.sum())))
    print("   class-centroid cosine to cluster:", sorted(sims, reverse=True)[:8])
    # nearest-neighbour labels of the cluster's tiles outside the cluster
    S = Z[cl == c0] @ Z[cl != c0].T; nb = np.argsort(-S, 1)[:, :5]; nl = lab[ii[cl != c0]][nb.ravel()]
    print("   5-NN (outside) labels:", np.bincount(nl, minlength=19).tolist())
