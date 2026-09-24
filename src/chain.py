"""Timeline reconstruction for one subject from shuffled 1-s windows.

Candidates per window a: union of top-K successors by tail3/head3 cosine, last/first cosine and mean-pool cosine.
Pair features (a -> b):
  s_th, s_11, s_mp            forward similarities
  s_th_rev                    cos(tail_b, head_a)  (should be LOWER than forward for a true successor)
  r_a, r_b                    s_th minus row max / column max (dominance)
  marg                        best-vs-second margin of a's row
  rank_a, rank_b              rank of b in a's row / of a in b's column (by s_th)
  same, gap, ext              same-limb flag, inertial boundary gap, linear-extrapolation gap (same limb only; NaN otherwise)
Scorer: LightGBM binary classifier fit on train sessions with known order.
Reconstruction: linear assignment on log-odds (one successor / one predecessor), drop edges below a threshold,
cut the weakest edge of each cycle -> chains.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

K_TH, K_11, K_MP = 40, 15, 15
FEAT_NAMES = ["s_th", "s_11", "s_mp", "s_th_rev", "r_a", "r_b", "marg", "rank_a", "rank_b", "same", "gap", "ext"]

def descriptors(V):
    V = np.asarray(V, np.float32)
    Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)
    def nrm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return dict(tail=nrm(Vn[:, -3:].mean(1)), head=nrm(Vn[:, :3].mean(1)), last=Vn[:, -1], first=Vn[:, 0], mp=nrm(Vn.mean(1)))

def _topk(S, k):
    return np.argpartition(-S, k, axis=1)[:, :k]

def pair_features(V, Xi, limb):
    """Returns cand (n,M) candidate successors per window (-1 padded) and feats (n,M,F) (NaN padded)."""
    n = len(V); D = descriptors(V)
    S_th = D["tail"] @ D["head"].T; np.fill_diagonal(S_th, -1)
    S_11 = D["last"] @ D["first"].T; np.fill_diagonal(S_11, -1)
    S_mp = D["mp"] @ D["mp"].T; np.fill_diagonal(S_mp, -1)
    c = np.concatenate([_topk(S_th, K_TH), _topk(S_11, K_11), _topk(S_mp, K_MP)], 1)
    # dedupe per row
    c.sort(1)
    dup = np.zeros_like(c, bool); dup[:, 1:] = c[:, 1:] == c[:, :-1]
    c[dup] = -1
    M = K_TH + K_11 + K_MP
    cand = np.full((n, M), -1, np.int64)
    for i in range(n):
        v = c[i][c[i] >= 0]; cand[i, :len(v)] = v
    valid = cand >= 0; cc = np.where(valid, cand, 0)
    rows = np.arange(n)[:, None]
    s_th = S_th[rows, cc]; s_11 = S_11[rows, cc]; s_mp = S_mp[rows, cc]; s_rev = S_th[cc, rows]
    rowmax = S_th.max(1); colmax = S_th.max(0)
    r_a = s_th - rowmax[:, None]; r_b = s_th - colmax[cc]
    srt = np.sort(S_th, 1); marg = (srt[:, -1] - srt[:, -2])[:, None].repeat(M, 1)
    # ranks (0 = best) by s_th
    order_r = np.argsort(-S_th, 1); rank_row = np.empty_like(order_r); rank_row[rows, order_r] = np.arange(n)[None, :]
    order_c = np.argsort(-S_th, 0); rank_col = np.empty_like(order_c); rank_col[order_c, np.arange(n)[None, :]] = np.arange(n)[:, None]
    rank_a = rank_row[rows, cc].astype(np.float32); rank_b = rank_col[rows, cc].astype(np.float32)
    limb = np.asarray(limb); same = (limb[:, None] == limb[cc]).astype(np.float32)
    Xi = np.asarray(Xi, np.float32)
    last = Xi[:, -1]; first = Xi[:, 0]; ext_pred = 2 * Xi[:, -1] - Xi[:, -2]
    gap = np.linalg.norm(last[:, None, :] - first[cc], axis=2); ext = np.linalg.norm(ext_pred[:, None, :] - first[cc], axis=2)
    gap = np.where(same > 0, np.log1p(gap), np.nan); ext = np.where(same > 0, np.log1p(ext), np.nan)
    feats = np.stack([s_th, s_11, s_mp, s_rev, r_a, r_b, marg, np.log1p(rank_a), np.log1p(rank_b), same, gap, ext], 2).astype(np.float32)
    feats[~valid] = np.nan
    return cand, feats

def random_limbs(n, rng):
    return rng.randint(0, 4, n)

def make_training_pairs(V, imu4, rng, limb=None):
    n = len(V)
    if limb is None: limb = random_limbs(n, rng)
    Xi = imu4[np.arange(n), limb]
    bad = np.isnan(Xi).any(axis=(1, 2))
    for i in np.where(bad)[0]:
        ok = [l for l in range(4) if not np.isnan(imu4[i, l]).any()]
        if ok: limb[i] = rng.choice(ok); Xi[i] = imu4[i, limb[i]]
    cand, feats = pair_features(V, Xi, limb)
    lab = (cand == (np.arange(n)[:, None] + 1)).astype(np.int64); lab[-1] = 0
    return feats, lab, cand, limb, Xi

class Scorer:
    def __init__(self, rounds=300):
        import lightgbm as lgb
        self.m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8)
    def fit(self, F, y):
        F = F.reshape(-1, F.shape[-1]); y = y.reshape(-1); ok = ~np.isnan(F[:, 0])
        self.m.fit(F[ok], y[ok]); return self
    def logodds(self, F):
        sh = F.shape[:-1]; Z = F.reshape(-1, F.shape[-1]); out = np.full(len(Z), -50.0, np.float32)
        ok = ~np.isnan(Z[:, 0])
        p = np.clip(self.m.predict_proba(Z[ok])[:, 1], 1e-6, 1 - 1e-6); out[ok] = np.log(p / (1 - p))
        return out.reshape(sh)

def assignment(cand, logodds, n):
    """Linear assignment; returns succ (n,) with every node assigned, and the edge log-odds sc (n,)."""
    L = np.full((n, n), -50.0, np.float32)
    valid = cand >= 0; rows = np.arange(n)[:, None]
    L[rows[valid[:, None].any(1)][:, 0:1].repeat(1, 1) if False else rows.repeat(cand.shape[1], 1)[valid], cand[valid]] = logodds[valid]
    np.fill_diagonal(L, -1e4)
    r, c = linear_sum_assignment(-L)
    return c.copy(), L[r, c].copy(), L

def cut(succ, sc, L, min_logodds):
    succ = succ.copy(); n = len(succ)
    succ[sc < min_logodds] = -1
    visited = np.zeros(n, bool)
    for st in range(n):
        if visited[st] or succ[st] < 0: continue
        path = []; x = st
        while x >= 0 and not visited[x]:
            visited[x] = True; path.append(x); x = succ[x]
        if x >= 0 and x in path:
            cyc = path[path.index(x):]
            w = np.array([L[a, succ[a]] for a in cyc]); a = cyc[int(w.argmin())]; succ[a] = -1
    return succ

def reconstruct(cand, logodds, n, min_logodds=0.0):
    succ, sc, L = assignment(cand, logodds, n)
    return cut(succ, sc, L, min_logodds)

def chains_from_succ(succ):
    n = len(succ); pred = np.full(n, -1); pred[succ[succ >= 0]] = np.where(succ >= 0)[0]
    out = []
    for st in np.where(pred < 0)[0]:
        ch = [int(st)]
        while succ[ch[-1]] >= 0: ch.append(int(succ[ch[-1]]))
        out.append(ch)
    return out

def edge_metrics(succ, n):
    e = succ >= 0
    correct = (succ[e] == np.arange(n)[e] + 1).sum()
    return dict(n_edges=int(e.sum()), precision=float(correct / max(e.sum(), 1)), recall=float(correct / (n - 1)))
