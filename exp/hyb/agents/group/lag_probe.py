"""Does a max-over-lags windowed cross-correlation (overlap>=25 samples) separate same-second / same-label pairs?"""
import sys, os, time, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import load_session, PAIRS, LIMBS, hungarian
from sklearn.metrics import roc_auc_score

name = sys.argv[1] if len(sys.argv) > 1 else "sbj_20"
HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
acc, lab = load_session(name); N = acc.shape[1]
acc = np.nan_to_num(acc)
mag = np.sqrt((acc ** 2).sum(-1))            # (4,N,50)
std = mag.std(-1); dyn = std > 0.08
MAXLAG = 25


def seg_norm(x):
    x = x - x.mean(-1, keepdims=True)
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)


def lagmax(A, B):
    """A,B: (N,50). max over lags |L|<=MAXLAG of Pearson corr on the overlap."""
    best = np.full((A.shape[0], B.shape[0]), -2.0, np.float32)
    for L in range(-MAXLAG, MAXLAG + 1):
        a = A[:, max(0, L):50 + min(0, L)]; b = B[:, max(0, -L):50 + min(0, -L)]
        c = seg_norm(a) @ seg_norm(b).T
        np.maximum(best, c, out=best)
    return best


rng = np.random.default_rng(0)
labs_u, inv = np.unique(lab, return_inverse=True)
sz = np.load(os.path.join(HERE, "scores", f"pair_sec_{name}.npz")) if os.path.exists(os.path.join(HERE, "scores", f"pair_sec_{name}.npz")) else None
for (a, b) in [(1, 3), (0, 2), (0, 1), (2, 3)]:
    t0 = time.time()
    M = lagmax(mag[a], mag[b])
    # also on the gravity-projected vertical component? keep magnitude only for speed
    i = np.arange(N)
    pos = M[i, i]
    jr = rng.integers(0, N, N); neg_r = M[i, jr]
    # same-label different-second negatives
    js = np.array([rng.choice(np.where(inv == inv[k])[0]) for k in range(N)]); neg_s = M[i, js]
    d = dyn[a] & dyn[b]
    y = np.r_[np.ones(N), np.zeros(N)]
    print(f"{name} {LIMBS[a]}-{LIMBS[b]} lagmax: AUC vs random {roc_auc_score(y, np.r_[pos, neg_r]):.3f} (dyn {roc_auc_score(np.r_[np.ones(d.sum()), np.zeros(d.sum())], np.r_[pos[d], neg_r[d]]):.3f}) | "
          f"vs same-label {roc_auc_score(y, np.r_[pos, neg_s]):.3f} (dyn {roc_auc_score(np.r_[np.ones(d.sum()), np.zeros(d.sum())], np.r_[pos[d], neg_s[d]]):.3f}) | "
          f"pos mean {pos.mean():.3f} dyn {pos[d].mean():.3f}; neg_r {neg_r.mean():.3f}; neg_s {neg_s.mean():.3f}  ({time.time()-t0:.0f}s)")
    # hungarian with lagmax alone and combined with cached LightGBM scores
    asg = hungarian(M); print(f"    lagmax alone: exact {np.mean(asg == i):.3f} same-label {np.mean(lab[asg] == lab):.3f} (dyn exact {np.mean((asg == i)[dyn[a]]):.3f})")
    if sz is not None:
        S = sz[f"S{min(a,b)}{max(a,b)}"].astype(np.float32)
        if a > b: S = S.T
        for alpha in (2.0, 5.0):
            asg = hungarian(S + alpha * M)
            print(f"    S + {alpha}*lagmax: exact {np.mean(asg == i):.3f} same-label {np.mean(lab[asg] == lab):.3f} (dyn exact {np.mean((asg == i)[dyn[a]]):.3f} same {np.mean((lab[asg] == lab)[dyn[a]]):.3f})")
        asg = hungarian(S); print(f"    S alone: exact {np.mean(asg == i):.3f} same-label {np.mean(lab[asg] == lab):.3f}")
