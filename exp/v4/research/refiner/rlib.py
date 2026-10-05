"""Vectorised boundary-refiner rows (same rows, same order, same 37 base features as v4_local.refiner_rows for H=3),
with a free window H and optional extra feature blocks, plus nested-CV helpers."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = 5


def load_cache():
    z = np.load(os.path.join(HERE, "cache", "fused.npz"), allow_pickle=True); d = {k: z[k] for k in z.files}
    s = np.load(os.path.join(HERE, "cache", "scalars.npz")); d.update({k: s[k] for k in s.files})
    return d


def chains(succ, depth):
    """nx[k][t] = k-th successor of t (-1 once broken), pv[k][t] = k-th predecessor"""
    n = len(succ); prv = np.full(n, -1, np.int64); m = succ >= 0; prv[succ[m]] = np.flatnonzero(m)
    nx, pv = [np.arange(n)], [np.arange(n)]
    for _ in range(depth):
        a = nx[-1]; nx.append(np.where(a >= 0, succ[np.maximum(a, 0)], -1))
        b = pv[-1]; pv.append(np.where(b >= 0, prv[np.maximum(b, 0)], -1))
    return nx, pv, prv


def vnorm_next(vmean, succ, chunk=8192):
    out = np.full(len(succ), -1.0, np.float32)
    for a in range(0, len(succ), chunk):
        s = succ[a:a + chunk]; m = s >= 0; ii = np.arange(a, a + len(s))[m]
        out[ii] = np.linalg.norm(vmean[ii] - vmean[s[m]], axis=1)
    return out


def take(A, idx, fill):
    """A[idx] with fill where idx < 0 (A 1-d)"""
    return np.where(idx >= 0, A[np.maximum(idx, 0)], fill)


def pair(M, idx, cur, oth):
    return M[idx, cur] - M[idx, oth]


def rows(fin, succ, B, lP, lQ, lw, sens, ener, post, vmot, vmean, H=3, dnext=None):
    """returns X (37 base features), dict of index arrays (g, cur, oth, off, i, j, nb-2..nb2)"""
    nx, pv, prv = chains(succ, H + 2)
    I = np.flatnonzero((succ >= 0) & (fin != fin[np.maximum(succ, 0)])); J = succ[I]
    offs = list(range(-H + 1, H + 1))
    G = np.stack([pv[-o][I] if o <= 0 else nx[o - 1][J] for o in offs], 1)            # (nb, 2H)
    OTH = np.stack([fin[J] if o <= 0 else fin[I] for o in offs], 1)
    OFF = np.broadcast_to(np.array(offs), G.shape)
    II, JJ = np.broadcast_to(I[:, None], G.shape), np.broadcast_to(J[:, None], G.shape)
    valid = (G >= 0)
    CUR = np.where(valid, fin[np.maximum(G, 0)], -1)
    valid &= CUR != OTH
    g, oth, cur, off, bi, bj = G[valid], OTH[valid], CUR[valid], OFF[valid], II[valid], JJ[valid]   # row-major = loop order
    if dnext is None:
        dnext = vnorm_next(vmean, succ)
    nxt, pr_ = succ[g], prv[g]
    base = [off.astype(np.float32), (cur == 0).astype(np.float32), (oth == 0).astype(np.float32),
            pair(B, g, cur, oth), pair(lP, g, cur, oth), pair(lw, g, cur, oth), pair(lQ, g, cur, oth),
            B[g, 0], lP[g, 0], ener[g], vmot[g], take(dnext, pr_, -1.0), np.where(nxt >= 0, dnext[g], -1.0)]
    nbs = {}
    for nb in (-2, -1, 1, 2):
        gj = pv[-nb][g] if nb < 0 else nx[nb][g]; ok = gj >= 0; gc = np.maximum(gj, 0); nbs[nb] = gj
        same = ok & (sens[gc] == sens[g])
        base += [np.where(ok, pair(B, gc, cur, oth), 0), np.where(ok, pair(lP, gc, cur, oth), 0), np.where(ok, ener[gc] - ener[g], 0),
                 same.astype(np.float32), np.where(same, np.linalg.norm(post[gc] - post[g], axis=1), -1.0),
                 np.where(ok, (fin[gc] == cur).astype(np.float32), -1.0)]
    X = np.stack([np.asarray(c, np.float32) for c in base], 1)
    return X, dict(g=g, cur=cur, oth=oth, off=off, i=bi, j=bj, nxt=nxt, prv=pr_, **{f"nb{k}": v for k, v in nbs.items()})


# ------------------------------------------------------------------ flips / scoring
def flips(fin, tiles, others, prob, K, thr=0.5, agg="sum"):
    """vectorised refiner_flips: per (tile, other) sum of p / K (agg='sum', the kernel) or mean over present rows
    ('mean'); flip tile to the best other when above thr"""
    n = len(fin); key = tiles * 32 + others
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, prob, len(uk))
    if agg == "sum":
        m = s / K
    else:
        m = s / np.bincount(inv, None, len(uk))
    gt, ot = uk // 32, uk % 32
    keep = m > thr
    gt, ot, m = gt[keep], ot[keep], m[keep]
    o = np.lexsort((-m, gt)); gt, ot, m = gt[o], ot[o], m[o]
    first = np.r_[True, gt[1:] != gt[:-1]]
    new = fin.copy(); new[gt[first]] = ot[first]
    return new, int(first.sum())
