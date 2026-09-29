"""path cover from successor links + sequence decoders"""
import numpy as np


def path_cover(succ, score, thr=-np.inf, mask=None):
    """keep link i->succ[i] if score>=thr; resolve in-degree conflicts by max score; break cycles at weakest edge.
    returns list of index arrays (paths in order), covering all rows (singletons included)."""
    n = len(succ)
    ok = (succ >= 0) & (score >= thr)
    if mask is not None:
        ok &= mask
    src = np.flatnonzero(ok); dst = succ[src]; sc = score[src]
    o = np.lexsort((-sc, dst))          # by dst, then score desc
    src, dst, sc = src[o], dst[o], sc[o]
    first = np.r_[True, dst[1:] != dst[:-1]]
    src, dst, sc = src[first], dst[first], sc[first]
    nxt = np.full(n, -1); nxt[src] = dst; ws = np.full(n, -np.inf); ws[src] = sc
    prv = np.full(n, -1); prv[dst] = src
    # self loops
    sl = np.flatnonzero(nxt == np.arange(n)); nxt[sl] = -1; prv[sl] = -1
    seen = np.zeros(n, bool); paths = []
    # cycles: nodes all with prv>=0 reachable only in cycles; handle by walking from heads first
    heads = np.flatnonzero(prv < 0)
    for h in heads:
        p = [h]; seen[h] = True; c = nxt[h]
        while c >= 0 and not seen[c]:
            p.append(c); seen[c] = True; c = nxt[c]
        paths.append(np.array(p))
    # remaining = cycles
    for s0 in np.flatnonzero(~seen):
        if seen[s0]:
            continue
        cyc = [s0]; seen[s0] = True; c = nxt[s0]
        while c >= 0 and not seen[c]:
            cyc.append(c); seen[c] = True; c = nxt[c]
        cyc = np.array(cyc); w = ws[cyc]; k = int(np.argmin(w))   # break after weakest edge cyc[k]->cyc[k+1]
        paths.append(np.r_[cyc[k + 1:], cyc[:k + 1]])
    return paths


def viterbi(L, lam):
    """L: (T,C) log emissions; uniform switch penalty lam. returns labels"""
    T, C = L.shape
    dp = L[0].copy(); bp = np.zeros((T, C), np.int64)
    for t in range(1, T):
        best = dp.argmax(); stay = dp; sw = dp[best] - lam
        take_sw = sw > stay
        bp[t] = np.where(take_sw, best, np.arange(C))
        dp = np.where(take_sw, sw, stay) + L[t]
    out = np.empty(T, np.int64); out[-1] = dp.argmax()
    for t in range(T - 1, 0, -1):
        out[t - 1] = bp[t, out[t]]
    return out


def fb_smooth(L, lam):
    """forward-backward posterior with sticky transition: stay prob 1-eps, switch eps/(C-1), eps=exp(-lam) scale.
    returns log posterior (T,C)"""
    T, C = L.shape
    eps = 1.0 / (1.0 + np.exp(lam)); A = np.full((C, C), eps / (C - 1)); np.fill_diagonal(A, 1 - eps)
    E = np.exp(L - L.max(1, keepdims=True))
    al = np.zeros((T, C)); be = np.ones((T, C))
    a = E[0] / E[0].sum(); al[0] = a
    for t in range(1, T):
        a = (a @ A) * E[t]; a /= a.sum(); al[t] = a
    b = np.ones(C)
    for t in range(T - 2, -1, -1):
        b = A @ (E[t + 1] * b); b /= b.sum(); be[t] = b
    post = al * be; post /= post.sum(1, keepdims=True)
    return np.log(np.clip(post, 1e-12, 1))
