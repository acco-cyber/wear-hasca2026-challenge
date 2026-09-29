"""Test-side diagnostic: do test subjects' windows split into their recording sessions in video space, and do those
clusters align with protocol blocks in our current predictions?  python test_session_diag.py <labels.csv>"""
import os, sys
import numpy as np, pandas as pd
KEEP = r"E:\Claude code\wear\work\hanbat\keep"
B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]
NSESS = {22: 3, 23: 3, 24: 2, 25: 2}
E = np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32)
sbj = np.load(os.path.join(KEEP, "blend.npz"))["test_sbj"].astype(int)
lab = pd.read_csv(sys.argv[1]).sort_values("id").target_feature.to_numpy()
rng = np.random.default_rng(0)

def kmeans(X, k, iters=100, restarts=10):
    best = None
    for r in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(2); a = d.argmin(1)
            C2 = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C): break
            C = C2
        inertia = d[np.arange(len(X)), a].sum()
        if best is None or inertia < best[0]: best = (inertia, a)
    return best[1]

for s in (22, 23, 24, 25):
    m = sbj == s; X = E[m]; X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6); X = X - X.mean(0)
    U, S_, Vt = np.linalg.svd(X, full_matrices=False); Z = U[:, :20] * S_[:20]
    l = lab[m]
    for k in sorted({2, NSESS[s]}):
        a = kmeans(Z, k)
        print(f"sbj {s} k={k}: sizes {np.bincount(a).tolist()}")
        for j in range(k):
            lj = l[a == j]; act = lj[lj > 0]
            b1 = np.isin(act, B1).mean() if len(act) else float("nan")
            cls = np.bincount(act, minlength=19)
            top = [(c, int(cls[c])) for c in np.argsort(-cls)[:10] if cls[c] > 0]
            print(f"   cluster {j}: n={len(lj)} null {np.mean(lj == 0):.2f} activity-in-B1 {b1:.2f}  top classes {top}")
