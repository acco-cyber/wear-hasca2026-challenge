"""find the visually distinct 1-activity sessions (sbj_22 / sbj_23 session_3, sbj_2 tail in train) and check link closure"""
from common import *
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

d = base(); ts = d["test_sbj"]
Vt = np.load(os.path.join(W, "data", "prep", "test_vid_mean768.npy")).astype(np.float32)
lab = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_labt.npy")).astype(int)
Qt = np.load(os.path.join(W, "subs", "sub_v4c_b4wa_Qt.npy"))
lk = np.load(os.path.join(K7, "links.npz")); tsucc = lk["test_succ"]; osucc = lk["oof_succ"]
st = np.load(os.path.join(K7, "stage.npz")); Bt = st["B2_TEST"]


def feats(X, ncomp=48, power=0.5):
    X = X - X.mean(0); p = PCA(ncomp, random_state=0).fit(X); Z = p.transform(X) / (p.explained_variance_ ** power + 1e-8)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def closure(ii, members, succ):
    """fraction of links leaving member tiles that land inside the member set (8 matchings)"""
    mem = np.zeros(succ.shape[1], bool); mem[members] = True
    out = succ[:, members]; ok = out >= 0
    return float(mem[out[ok]].mean())


for s, ks in ((22, (3, 4, 6, 8, 10)), (23, (3, 6, 8, 10))):
    ii = np.flatnonzero(ts == s); Z = feats(Vt[ii])
    for k in ks:
        cl = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="discretize").fit_predict(Z)
        for c in range(k):
            jj = ii[cl == c]
            if len(jj) > 600:
                continue
            cnt = np.bincount(lab[jj], minlength=19); top = [(int(a), int(cnt[a])) for a in np.argsort(cnt)[::-1][:3] if cnt[a] > 0]
            # mean cosine to own cluster vs to rest of subject (raw mean-pooled video)
            U = Vt[ii] / np.linalg.norm(Vt[ii], axis=1, keepdims=True); m_in = U[cl == c].mean(0); m_out = U[cl != c].mean(0)
            print(f"sbj {s} k={k} cluster {c}: n={len(jj)} top {top} link-closure {closure(ii, jj, tsucc):.3f} "
                  f"cos(in,out centroid) {m_in @ m_out / np.linalg.norm(m_in) / np.linalg.norm(m_out):.3f}  B2-argmax {np.bincount(Bt[jj].argmax(1), minlength=19).argsort()[::-1][:3].tolist()}")
# training analogue: sbj_2 tail and sbj_10 tail closure under OOF links
for r, t0 in ((14, 3445), (3, 2564)):
    ii = np.flatnonzero(d["oof_rec"] == r); jj = ii[d["t"][ii] >= t0]
    print(f"train rec {r} tail ({len(jj)} tiles): link closure {closure(ii, jj, osucc):.3f}")
# baseline closure of a random contiguous 100-s block in train
rng = np.random.default_rng(0); cl_ = []
for _ in range(50):
    r = rng.integers(24); ii = ordered(d, r); a = rng.integers(len(ii) - 120); cl_.append(closure(ii, ii[a:a + 107], osucc))
print("random contiguous 107-s train segment closure", np.mean(cl_).round(3))
