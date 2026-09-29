"""Task 1b/2: per-subject hierarchical clustering (scipy) of windows into bout-like clusters; ceilings + practical decoders.
python clus_ceil.py <dist-variant> ; saves linkage matrices to Z_<variant>.npz"""
import sys, time, os
import numpy as np, scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import *

var = sys.argv[1] if len(sys.argv) > 1 else "avg_vp"
d = load_oof()
y, rec, st, sbj, fold = d["y"], d["rec"], d["start"], d["sbj"], d["fold"]
P = d["P"]; n = len(y); succ = d["succ"]; score = d["score"]
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
Q = cal_Q(P, sbj, TRAIN_SETS); lq = np.log(np.clip(Q, 1e-9, 1))
bid = true_bouts(y, rec, st)
E = load_emb("oof"); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
base = macro_f1(y, pred)
print("baseline", round(base, 4), "acc", round(np.mean(pred == y), 4))
for s in np.unique(sbj)[:5]:
    m = sbj == s; print("sbj", s, "true bouts", len(np.unique(bid[m])), "activity bouts>=10", sum(1 for b in np.unique(bid[m]) if y[bid == b][0] > 0 and (bid == b).sum() >= 10))

fn = f"Z_{var}.npz"
if os.path.exists(fn):
    Zs = dict(np.load(fn))
else:
    Zs = {}
    for s in np.unique(sbj):
        t0 = time.time(); ii = np.flatnonzero(sbj == s); m = len(ii)
        S = E[ii] @ E[ii].T
        if "vp" in var:
            Qs = np.sqrt(P[ii]).astype(np.float32); S = S + 0.5 * (Qs @ Qs.T); D = 1.5 - S
        elif "vq" in var:   # calibrated probs
            Qs = np.sqrt(Q[ii]).astype(np.float32); S = S + 0.5 * (Qs @ Qs.T); D = 1.5 - S
        else:
            D = 1.0 - S
        if "lk" in var:   # links shrink distance of linked pairs
            loc = -np.ones(n, np.int64); loc[ii] = np.arange(m); mm = succ[ii] >= 0
            a_ = np.flatnonzero(mm); b_ = loc[succ[ii][mm]]; w = 1 / (1 + np.exp(-score[ii][mm]))
            f = 1 - 0.5 * w
            D[a_, b_] *= f; D[b_, a_] *= f
        D = np.maximum(D, 0).astype(np.float64); np.fill_diagonal(D, 0); D = (D + D.T) / 2
        method = "average" if var.startswith("avg") else ("complete" if var.startswith("cmp") else "weighted")
        Zs[str(s)] = linkage(squareform(D, checks=False), method=method)
        print(f"sbj {s} n={m} linkage {time.time() - t0:.1f}s", flush=True)
    np.savez(fn, **Zs)


def decode(labels_c, how, thr=0.0):
    """labels_c: cluster id per row (global unique). returns predictions"""
    out = pred.copy()
    u, inv = np.unique(labels_c, return_inverse=True)
    if how == "true":
        cnt = np.zeros((len(u), N_CLS)); np.add.at(cnt, (inv, y), 1); return cnt.argmax(1)[inv]
    cnt = np.zeros((len(u), N_CLS)); np.add.at(cnt, (inv, pred), 1)
    if how == "maj":
        lab = cnt.argmax(1)
    elif how == "logq":
        sq = np.zeros((len(u), N_CLS)); np.add.at(sq, inv, lq); lab = sq.argmax(1)
    elif how == "meanq":
        sq = np.zeros((len(u), N_CLS)); np.add.at(sq, inv, Q); lab = sq.argmax(1)
    pur = cnt.max(1) / cnt.sum(1); size = cnt.sum(1)
    use = (pur >= thr)[inv]
    out[use] = lab[inv][use]
    return out


print("variant", var)
for K in (20, 30, 45, 60, 90, 130, 200, 300, 500):
    lab_c = np.zeros(n, np.int64); off = 0
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); c = fcluster(Zs[str(s)], t=K, criterion="maxclust"); lab_c[ii] = c + off; off += c.max() + 1
    nclu = len(np.unique(lab_c))
    # cluster/bout purity: fraction of rows whose cluster majority true bout == their bout
    ceil = macro_f1(y, decode(lab_c, "true"))
    r = [f"K={K:3d} ({nclu} clusters) TRUE-maj {ceil:.4f}"]
    for how in ("maj", "logq", "meanq"):
        r.append(f"{how} {macro_f1(y, decode(lab_c, how)):.4f}")
    for thr in (0.6, 0.8):
        r.append(f"maj@{thr} {macro_f1(y, decode(lab_c, 'maj', thr)):.4f}")
    print(" | ".join(r), flush=True)
