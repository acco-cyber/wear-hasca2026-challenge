"""Pair features v2: old chain.py descriptors + contrastive encoder scores (row/col softmax, ranks, reverse)
+ candidates from several descriptors (row and column top-K)."""
from lk import *

def nrm(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def ranks(S):
    """rank_row[i,j] = rank of j in row i (0 best); rank_col[i,j] = rank of i in column j."""
    n = len(S); rows = np.arange(n)[:, None]
    o = np.argsort(-S, 1); rr = np.empty((n, n), np.int32); rr[rows, o] = np.arange(n, dtype=np.int32)[None, :]
    o = np.argsort(-S, 0); rc = np.empty((n, n), np.int32); rc[o, np.arange(n)[None, :]] = np.arange(n, dtype=np.int32)[:, None]
    return rr, rc

def lse(S, axis):
    m = S.max(axis, keepdims=True); return (m + np.log(np.exp(S - m).sum(axis, keepdims=True)))

FEAT2 = ["s_th", "s_11", "s_mp", "s_th_rev", "r_a", "r_b", "marg", "rank_a", "rank_b", "same", "gap", "ext",
         "nn", "nn_rev", "nn_prow", "nn_pcol", "nn_rank_a", "nn_rank_b", "nn_marg", "nn_2hop", "imu_e", "imu_de", "grav_cos"]

def pair_features2(V, Xi, limb, Snn, K_NN=40, K_NNC=20, K_TH=20, K_11=10):
    n = len(V); V = np.asarray(V, np.float32); Vn = nrm(V)
    tail = nrm(Vn[:, -3:].mean(1)); head = nrm(Vn[:, :3].mean(1)); mp = nrm(Vn.mean(1))
    S_th = tail @ head.T; np.fill_diagonal(S_th, -1)
    S_11 = Vn[:, -1] @ Vn[:, 0].T; np.fill_diagonal(S_11, -1)
    S_mp = mp @ mp.T; np.fill_diagonal(S_mp, -1)
    Snn = Snn.astype(np.float32).copy(); np.fill_diagonal(Snn, -1e4)
    rr_nn, rc_nn = ranks(Snn)
    def topk(S, k): return np.argpartition(-S, k, axis=1)[:, :k]
    c = np.concatenate([topk(Snn, K_NN), topk(S_th, K_TH), topk(S_11, K_11)], 1)
    # column top-K: for each b, its K_NNC best predecessors a -> add b to cand[a]
    colc = topk(Snn.T, K_NNC)   # (n_b, K) predecessors
    extra = [[] for _ in range(n)]
    for bb in range(n):
        for aa in colc[bb]: extra[aa].append(bb)
    M = c.shape[1] + K_NNC * 3
    cand = np.full((n, M), -1, np.int64)
    for i in range(n):
        v = np.unique(np.concatenate([c[i], np.array(extra[i], np.int64)])); v = v[v != i][:M]; cand[i, :len(v)] = v
    Mreal = int((cand >= 0).sum(1).max()); cand = cand[:, :Mreal]
    valid = cand >= 0; cc = np.where(valid, cand, 0); rows = np.arange(n)[:, None]
    s_th = S_th[rows, cc]; s_11 = S_11[rows, cc]; s_mp = S_mp[rows, cc]; s_rev = S_th[cc, rows]
    r_a = s_th - S_th.max(1)[:, None]; r_b = s_th - S_th.max(0)[cc]
    srt = np.sort(S_th, 1); marg = (srt[:, -1] - srt[:, -2])[:, None].repeat(Mreal, 1)
    rr, rc = ranks(S_th); rank_a = rr[rows, cc].astype(np.float32); rank_b = rc[rows, cc].astype(np.float32); del rr, rc
    limb = np.asarray(limb); same = (limb[:, None] == limb[cc]).astype(np.float32)
    Xi = np.nan_to_num(np.asarray(Xi, np.float32))
    last = Xi[:, -1]; first = Xi[:, 0]; ext_pred = 2 * Xi[:, -1] - Xi[:, -2]
    gap = np.linalg.norm(last[:, None, :] - first[cc], axis=2); ext = np.linalg.norm(ext_pred[:, None, :] - first[cc], axis=2)
    gap = np.where(same > 0, np.log1p(gap), np.nan); ext = np.where(same > 0, np.log1p(ext), np.nan)
    # encoder features
    nn_ = Snn[rows, cc]; nn_rev = Snn[cc, rows]
    prow = Snn - lse(Snn, 1); pcol = Snn - lse(Snn, 0)
    nn_prow = prow[rows, cc]; nn_pcol = pcol[rows, cc]
    nn_rank_a = np.log1p(rr_nn[rows, cc].astype(np.float32)); nn_rank_b = np.log1p(rc_nn[rows, cc].astype(np.float32))
    srt = np.sort(Snn, 1); nn_marg = nn_ - srt[:, -1][:, None]
    # 2-hop: best continuation b->c score (is b itself a good predecessor of something?) + a's best predecessor
    nn_2hop = (prow.max(1)[cc] + pcol.max(0)[:, None])
    # IMU activity-level features (limb-agnostic): energy + its change
    mag = np.linalg.norm(Xi, axis=2); e = np.log1p(mag.std(1)); imu_e = np.abs(e[:, None] - e[cc]); imu_de = e[cc] - e[:, None]
    g = nrm(Xi.mean(1)); grav_cos = np.where(same > 0, (g[:, None, :] * g[cc]).sum(2), np.nan)
    feats = np.stack([s_th, s_11, s_mp, s_rev, r_a, r_b, marg, np.log1p(rank_a), np.log1p(rank_b), same, gap, ext,
                      nn_, nn_rev, nn_prow, nn_pcol, nn_rank_a, nn_rank_b, nn_marg, nn_2hop, imu_e, imu_de, grav_cos], 2).astype(np.float32)
    feats[~valid] = np.nan
    return cand, feats
