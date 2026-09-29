"""Task 1a: ceilings of label-free neighbourhood signals on OOF rows.
For each window: majority of TRUE labels of its k video-NN (within subject), link-hop neighbourhoods;
also majority of PREDICTED labels (practical) and same-bout fraction."""
import numpy as np, time
from common import *

d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y)
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS)
bid = true_bouts(y, rec, st)
E0 = load_emb("oof")
base = macro_f1(y, pred)
print("baseline", round(base, 4))


def norm(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)


def centred(E):
    Ec = E.copy()
    for s in np.unique(sbj):
        m = sbj == s; Ec[m] -= Ec[m].mean(0)
    return Ec


# link-smoothed embedding (as hanbat smooth_emb g=0.5)
succ = d["succ"]; score = d["score"]
import scipy.sparse as sp
ms = succ >= 0; wl = 1 / (1 + np.exp(-score[ms]))
Lw = sp.csr_matrix((np.r_[wl, wl], (np.r_[np.flatnonzero(ms), succ[ms]], np.r_[succ[ms], np.flatnonzero(ms)])), shape=(n, n))
deg = np.asarray(Lw.sum(1)).ravel(); deg[deg == 0] = 1
An = sp.diags(1 / deg) @ Lw

variants = {}
En = norm(E0); variants["raw"] = En
Ec = norm(centred(E0)); variants["centred"] = Ec
variants["raw+link0.5"] = norm(En + 0.5 * (An @ En))
variants["centred+link0.5"] = norm(Ec + 0.5 * (An @ Ec))
KMAX = 20
res = {}
for name, E in variants.items():
    for use_p in (0.0, 0.5):
        t0 = time.time()
        nb = np.zeros((n, KMAX), np.int64)
        for s in np.unique(sbj):
            ii = np.flatnonzero(sbj == s); S = E[ii] @ E[ii].T
            if use_p:
                Qs = np.sqrt(P[ii]).astype(np.float32); S = S + use_p * (Qs @ Qs.T)
            np.fill_diagonal(S, -np.inf)
            k_ = np.argpartition(-S, KMAX, axis=1)[:, :KMAX]; v = np.take_along_axis(S, k_, 1)
            o = np.argsort(-v, 1); nb[ii] = ii[np.take_along_axis(k_, o, 1)]
        np.save(f"nb_{name}_p{use_p}.npy", nb)
        for k in (5, 10, 20):
            nk = nb[:, :k]
            same_b = np.mean(bid[nk] == bid[:, None]); same_y = np.mean(y[nk] == y[:, None])
            # majority of true labels incl self vote
            votes = np.zeros((n, N_CLS)); np.add.at(votes, (np.repeat(np.arange(n), k), y[nk].ravel()), 1)
            vt = votes.copy(); vt[np.arange(n), y] += 1
            ceil = macro_f1(y, vt.argmax(1))
            # majority of predicted labels incl self
            vp = np.zeros((n, N_CLS)); np.add.at(vp, (np.repeat(np.arange(n), k), pred[nk].ravel()), 1); vp[np.arange(n), pred] += 1
            prac = macro_f1(y, vp.argmax(1))
            # summed log Q
            lq = np.log(np.clip(Q, 1e-9, 1)); sq = lq[nk].sum(1) + lq
            prac2 = macro_f1(y, sq.argmax(1))
            print(f"{name:16s} p={use_p} k={k:2d}: same-bout {same_b:.3f} same-label {same_y:.3f} | TRUE-maj F1 {ceil:.4f} | pred-maj F1 {prac:.4f} | sum-logQ F1 {prac2:.4f}  ({time.time() - t0:.0f}s)", flush=True)

# link hops
prv = np.full(n, -1); prv[succ[ms]] = np.flatnonzero(ms)
for H in (1, 2, 3):
    nbrs = []; fw = np.arange(n); bw = np.arange(n)
    for h in range(H):
        fw = np.where(fw >= 0, succ[np.maximum(fw, 0)], -1); bw = np.where(bw >= 0, prv[np.maximum(bw, 0)], -1)
        nbrs += [fw.copy(), bw.copy()]
    NB = np.stack(nbrs, 1); valid = NB >= 0
    same_b = np.mean((bid[np.maximum(NB, 0)] == bid[:, None])[valid]); same_y = np.mean((y[np.maximum(NB, 0)] == y[:, None])[valid])
    votes = np.zeros((n, N_CLS)); rr = np.repeat(np.arange(n), NB.shape[1])[valid.ravel()]
    np.add.at(votes, (rr, y[NB.ravel()[valid.ravel()]]), 1); votes[np.arange(n), y] += 1
    vp = np.zeros((n, N_CLS)); np.add.at(vp, (rr, pred[NB.ravel()[valid.ravel()]]), 1); vp[np.arange(n), pred] += 1.01
    print(f"link hops {H}: mean nbrs {valid.sum(1).mean():.2f} same-bout {same_b:.3f} same-label {same_y:.3f} | TRUE-maj F1 {macro_f1(y, votes.argmax(1)):.4f} | pred-maj F1 {macro_f1(y, vp.argmax(1)):.4f}")
