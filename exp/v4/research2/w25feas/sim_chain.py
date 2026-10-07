"""(a) Same-limb sample continuity alone: per subject (sessions pooled, as in the 2025 file which has no session id) and
per limb, all 1-s tiles are shuffled; a continuity cost between the tail of tile i and the head of tile j is computed for
every pair; Hungarian 1:1 assignment + cycle breaking (the kernel's lk_assign rule) gives one successor per tile.
Saves per (subject, limb): predicted successor, its cost, the top-K candidate list with costs.
  python sim_chain.py --cost nll [--K 30] [--subjects 5,9]"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--cost", default="nll", choices=["raw", "nll", "nll3"]); ap.add_argument("--K", type=int, default=30)
ap.add_argument("--subjects", default=""); ap.add_argument("--jobs", type=int, default=3)


def boundary_stats(X):
    """X (n,50,3): tail/head samples and a local noise scale from second differences near each edge"""
    d2t = np.diff(X[:, -10:], 2, axis=1); d2h = np.diff(X[:, :10], 2, axis=1)
    st = (d2t ** 2).sum(2).mean(1) + 1e-5; sh = (d2h ** 2).sum(2).mean(1) + 1e-5
    return X[:, -1], X[:, -2], X[:, -3], X[:, 0], X[:, 1], X[:, 2], st, sh


def cost_matrix(X, kind):
    T0, T1, T2, H0, H1, H2, st, sh = boundary_stats(X.astype(np.float64)); n = len(X)
    C = np.empty((n, n), np.float64)
    for b in range(0, n, 256):
        sl = slice(b, min(n, b + 256))
        fwd = (2 * T0[sl] - T1[sl])[:, None, :]                       # forward extrapolation of the tail
        bwd = (2 * H0 - H1)[None, :, :]                               # backward extrapolation of the head
        r1 = ((fwd - H0[None]) ** 2).sum(2); r2 = ((T0[sl][:, None, :] - bwd) ** 2).sum(2); r0 = ((T0[sl][:, None, :] - H0[None]) ** 2).sum(2)
        if kind == "raw":
            C[sl] = r0 + 0.5 * r1 + 0.5 * r2
        else:
            v = st[sl][:, None] + sh[None, :]                           # noise of a 1-step prediction ~ local 2nd-difference power
            c = (r1 + r2) / v + 3.0 * np.log(v)
            if kind == "nll3":                                          # add a quadratic (3-sample) extrapolation in both directions
                fq = (3 * T0[sl] - 3 * T1[sl] + T2[sl])[:, None, :]; bq = (3 * H0 - 3 * H1 + H2)[None, :, :]
                c = c + 0.5 * (((fq - H0[None]) ** 2).sum(2) + ((T0[sl][:, None, :] - bq) ** 2).sum(2)) / (2 * v)
            C[sl] = c
    np.fill_diagonal(C, np.inf)
    return C


def assign(C):
    """Hungarian on a dense cost, then break every cycle at its most expensive link"""
    n = len(C); Cf = np.where(np.isfinite(C), C, 1e12)
    r, c = linear_sum_assignment(Cf); succ = np.full(n, -1, np.int64); succ[r] = c
    seen = np.zeros(n, bool)
    for s in range(n):
        if seen[s]:
            continue
        path, pos, node = [], {}, s
        while node >= 0 and not seen[node]:
            seen[node] = True; pos[node] = len(path); path.append(node); node = int(succ[node])
        if node >= 0 and node in pos:
            cyc = path[pos[node]:]; succ[cyc[int(np.argmax([C[x, succ[x]] for x in cyc]))]] = -1
    return succ


def run_subject(S, s, kind, K):
    t0 = time.time(); ii, T = subject_tiles(S, s); n = len(ii)
    pos = np.full(len(S["oof_y"]), -1); pos[ii] = np.arange(n)
    ts = S["true_succ"][ii]; tl = np.where(ts >= 0, pos[np.maximum(ts, 0)], -1)
    out = dict(rows=ii, true_loc=tl)
    for L in range(4):
        C = cost_matrix(T[L], kind)
        succ = assign(C)
        top = np.argsort(C, 1)[:, :K]
        out[f"succ{L}"] = succ; out[f"cost{L}"] = np.where(succ >= 0, C[np.arange(n), np.maximum(succ, 0)], np.nan)
        out[f"top{L}"] = top.astype(np.int32); out[f"topc{L}"] = np.take_along_axis(C, top, 1).astype(np.float32)
        h = tl >= 0; tc = C[np.flatnonzero(h), tl[h]]
        out[f"rank{L}"] = np.full(n, -1); out[f"rank{L}"][h] = (C[h] < tc[:, None]).sum(1)
    np.savez(os.path.join(OUT, "cache", f"chain_{kind}_s{s}.npz"), **out)
    msg = " ".join(f"L{L} {np.mean(out[f'succ{L}'][tl >= 0] == tl[tl >= 0]):.3f}" for L in range(4))
    return f"sbj {s} n={n}: exact {msg} [{time.time() - t0:.0f}s]"


if __name__ == "__main__":
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    S = load_stage(); subs = [int(x) for x in a.subjects.split(",")] if a.subjects else [int(x) for x in np.unique(S["oof_sbj"])]
    subs = sorted(subs, key=lambda s: -(S["oof_sbj"] == s).sum())
    for m in Parallel(n_jobs=a.jobs)(delayed(run_subject)(S, s, a.cost, a.K) for s in subs):
        print(m, flush=True)
