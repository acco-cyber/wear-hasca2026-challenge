"""Decoding of per-window class probabilities along reconstructed chains."""
import numpy as np

NC = 19

def viterbi_chains(P, chains, p_stay=0.95, temp=1.0, prior=None):
    """P (n,NC) probabilities; chains: list of ordered index lists. Returns labels (n,)."""
    n = len(P); out = P.argmax(1).copy()
    logE = np.log(np.clip(P, 1e-6, 1)) / temp
    if prior is not None: logE = logE + np.log(prior)[None, :]
    logT = np.full((NC, NC), np.log((1 - p_stay) / (NC - 1))); np.fill_diagonal(logT, np.log(p_stay))
    for ch in chains:
        if len(ch) == 1: continue
        E = logE[ch]; T = len(ch)
        delta = E[0].copy(); back = np.zeros((T, NC), np.int64)
        for t in range(1, T):
            m = delta[:, None] + logT            # (from, to)
            back[t] = m.argmax(0); delta = m.max(0) + E[t]
        lab = np.zeros(T, np.int64); lab[-1] = delta.argmax()
        for t in range(T - 1, 0, -1): lab[t - 1] = back[t, lab[t]]
        out[ch] = lab
    return out

def chain_vote(P, chains):
    out = P.argmax(1).copy(); logP = np.log(np.clip(P, 1e-6, 1))
    for ch in chains:
        if len(ch) > 1: out[ch] = logP[ch].sum(0).argmax()
    return out

def calibrate_counts(P, chains, lo=40, hi=230, p_stay=0.95, temp=1.0, iters=40, step=0.25, verbose=False):
    """Per-subject decoding with additive log-biases adjusted so that every activity class (1..18)
    gets between lo and hi windows after Viterbi decoding. Returns labels and the bias vector."""
    n = len(P); b = np.zeros(NC)
    logP = np.log(np.clip(P, 1e-6, 1))
    for it in range(iters):
        Pb = np.exp(logP + b[None, :]); Pb /= Pb.sum(1, keepdims=True)
        lab = viterbi_chains(Pb, chains, p_stay=p_stay, temp=temp)
        cnt = np.bincount(lab, minlength=NC)
        under = np.where(cnt[1:] < lo)[0] + 1; over = np.where(cnt[1:] > hi)[0] + 1
        if verbose: print(f"  iter {it}: null {cnt[0]/n:.3f} under {under.tolist()} over {over.tolist()}")
        if len(under) == 0 and len(over) == 0: break
        b[under] += step; b[over] -= step
    return lab, b

def build_graph(cand, logodds, n, k=10, min_logodds=-8.0):
    """Symmetric weighted neighbour lists from successor candidates: for each window keep the top-k successors and
    top-k predecessors by log-odds. Returns list of (idx array, weight array) per node; weight = sigmoid(logodds)."""
    succ_n = [[] for _ in range(n)]; pred_n = [[] for _ in range(n)]
    valid = cand >= 0
    for a in range(n):
        m = valid[a]; c = cand[a][m]; lo = logodds[a][m]
        top = np.argsort(-lo)[:k]
        for j in top:
            if lo[j] < min_logodds: continue
            succ_n[a].append((int(c[j]), float(lo[j]))); pred_n[int(c[j])].append((a, float(lo[j])))
    out = []
    for a in range(n):
        pr = sorted(pred_n[a], key=lambda x: -x[1])[:k]
        allv = succ_n[a] + pr
        if not allv: out.append((np.zeros(0, np.int64), np.zeros(0))); continue
        idx = np.array([v[0] for v in allv]); w = 1 / (1 + np.exp(-np.array([v[1] for v in allv])))
        out.append((idx, w))
    return out

def graph_smooth(P, graph, alpha=0.5, iters=5, self_w=1.0):
    """Iterative propagation of log-probabilities over the neighbour graph. Returns smoothed probabilities."""
    logP0 = np.log(np.clip(P, 1e-6, 1)); logP = logP0.copy()
    for _ in range(iters):
        new = logP0.copy()
        for a, (idx, w) in enumerate(graph):
            if len(idx) == 0: continue
            nb = (w[:, None] * logP[idx]).sum(0) / (w.sum() + 1e-9)
            new[a] = (1 - alpha) * logP0[a] + alpha * nb
        logP = new
    out = np.exp(logP - logP.max(1, keepdims=True)); out /= out.sum(1, keepdims=True)
    return out

def segments(labels, chains):
    """Runs of equal labels along chains -> list of (label, [indices])."""
    segs = []
    for ch in chains:
        cur = [ch[0]]
        for i in ch[1:]:
            if labels[i] == labels[cur[-1]]: cur.append(i)
            else: segs.append((int(labels[cur[0]]), cur)); cur = [i]
        segs.append((int(labels[cur[0]]), cur))
    return segs
