"""(a, learned) candidate pairs + features for a per-limb continuity scorer. Per subject and limb: dense nll3 cost,
candidates = top-K successors of every tile by cost U top-Kc predecessors of every tile; features from the tile edges
(1- and 2-step extrapolations, low-pass trend, periodic continuation, pose/energy change, ranks).
  python chain_feats.py [--K 25] [--Kc 10]  ->  cache/feat_s{s}.npz"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
from common import *
from sim_chain import cost_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=25); ap.add_argument("--Kc", type=int, default=10)
ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--subjects", default="")
a = ap.parse_args()


def tile_stats(X):
    X = X.astype(np.float64); n = len(X)
    d2t = (np.diff(X[:, -10:], 2, axis=1) ** 2).sum(2).mean(1) + 1e-5; d2h = (np.diff(X[:, :10], 2, axis=1) ** 2).sum(2).mean(1) + 1e-5
    t = np.arange(10) - 4.5
    sl_t = (X[:, -10:] * t[None, :, None]).sum(1) / (t ** 2).sum(); mu_t = X[:, -10:].mean(1)
    sl_h = (X[:, :10] * t[None, :, None]).sum(1) / (t ** 2).sum(); mu_h = X[:, :10].mean(1)
    pred_f = mu_t + sl_t * (4.5 + 3.0)        # trend of the last 10 samples evaluated at the mean position of the next 5
    pred_b = mu_h - sl_h * (4.5 + 3.0)        # trend of the first 10 samples evaluated at the mean position of the previous 5
    m5h = X[:, :5].mean(1); m5t = X[:, -5:].mean(1)
    res_t = ((X[:, -10:] - (mu_t[:, None] + sl_t[:, None] * t[None, :, None])) ** 2).sum(2).mean(1) + 1e-5
    res_h = ((X[:, :10] - (mu_h[:, None] + sl_h[:, None] * t[None, :, None])) ** 2).sum(2).mean(1) + 1e-5
    # dominant period from the autocorrelation of the demeaned signal (lags 8..40)
    Z = X - X.mean(1, keepdims=True); ac = np.stack([(Z[:, lag:] * Z[:, :-lag]).sum((1, 2)) / (50 - lag) for lag in range(8, 41)], 1)
    var = (Z ** 2).sum((1, 2)) / 50 + 1e-6; P = 8 + ac.argmax(1); acmax = ac.max(1) / var
    per_f = np.stack([X[np.arange(n), 50 - P + k] for k in range(5)], 1)       # predicted next 5 samples (one period back)
    per_b = np.stack([X[np.arange(n), P - 5 + k] for k in range(5)], 1)        # predicted previous 5 samples (one period ahead)
    mag = np.linalg.norm(X, axis=2)
    return dict(T0=X[:, -1], T1=X[:, -2], T2=X[:, -3], H0=X[:, 0], H1=X[:, 1], H2=X[:, 2], st=d2t, sh=d2h, pf=pred_f, pb=pred_b,
                m5h=m5h, m5t=m5t, rt=res_t, rh=res_h, perf=per_f, perb=per_b, head5=X[:, :5], tail5=X[:, -5:], acm=acmax, var=var,
                mean=X.mean(1), lstd=np.log(mag.std(1) + 1e-4), P=P.astype(np.float64))


FEATS = ["c_nll3", "rank_row", "rank_col", "d_rowbest", "d_colbest", "d_row2", "r1", "r2", "rq1", "rq2", "r0", "logv", "dlog_noise",
         "lp_f", "lp_b", "lp_mid", "per_f", "per_b", "acm_i", "acm_j", "dP", "pose", "dlstd", "lstd_i", "lstd_j", "limb"]


def pair_feats(F, i, j, C, rank_row, rank_col, rowbest, row2, colbest, L):
    v = F["st"][i] + F["sh"][j]
    fwd = 2 * F["T0"][i] - F["T1"][i]; bwd = 2 * F["H0"][j] - F["H1"][j]
    fq = 3 * F["T0"][i] - 3 * F["T1"][i] + F["T2"][i]; bq = 3 * F["H0"][j] - 3 * F["H1"][j] + F["H2"][j]
    sq = lambda d: (d ** 2).sum(-1)
    vl = F["rt"][i] / 5 + F["rh"][j] / 5 + 1e-5
    out = np.stack([
        C[i, j], rank_row, rank_col, C[i, j] - rowbest[i], C[i, j] - colbest[j], C[i, j] - row2[i],
        sq(fwd - F["H0"][j]) / v, sq(F["T0"][i] - bwd) / v, sq(fq - F["H0"][j]) / v, sq(F["T0"][i] - bq) / v, sq(F["T0"][i] - F["H0"][j]) / v,
        np.log(v), np.abs(np.log(F["st"][i]) - np.log(F["sh"][j])),
        sq(F["pf"][i] - F["m5h"][j]) / vl, sq(F["pb"][j] - F["m5t"][i]) / vl, sq(F["m5t"][i] - F["m5h"][j]) / vl,
        sq(F["perf"][i] - F["head5"][j]).sum(1) / (5 * (F["var"][i] + F["var"][j])), sq(F["perb"][j] - F["tail5"][i]).sum(1) / (5 * (F["var"][i] + F["var"][j])),
        F["acm"][i], F["acm"][j], np.abs(F["P"][i] - F["P"][j]), sq(F["mean"][i] - F["mean"][j]), np.abs(F["lstd"][i] - F["lstd"][j]),
        F["lstd"][i], F["lstd"][j], np.full(len(i), L, np.float64)], 1)
    return out.astype(np.float32)


def run_subject(S, s, K, Kc):
    t0 = time.time(); ii, T = subject_tiles(S, s); n = len(ii)
    pos = np.full(len(S["oof_y"]), -1); pos[ii] = np.arange(n)
    ts = S["true_succ"][ii]; tl = np.where(ts >= 0, pos[np.maximum(ts, 0)], -1)
    out = dict(rows=ii, true_loc=tl); recall = []
    for L in range(4):
        C = cost_matrix(T[L], "nll3"); F = tile_stats(T[L])
        top = np.argpartition(C, K, 1)[:, :K]
        topc = np.argpartition(C, Kc, 0)[:Kc, :]                                   # best predecessors of every column
        pi = np.r_[np.repeat(np.arange(n), K), topc.ravel()]; pj = np.r_[top.ravel(), np.tile(np.arange(n), Kc)]
        key = np.unique(pi.astype(np.int64) * n + pj); pi, pj = key // n, key % n
        ok = pi != pj; pi, pj = pi[ok], pj[ok]
        cv = C[pi, pj]; rr = np.zeros(len(pi)); rc = np.zeros(len(pi))
        CT = np.ascontiguousarray(C.T)
        for b in range(0, n, 64):                                                  # ranks of each candidate in its row / column
            m = np.flatnonzero((pi >= b) & (pi < b + 64)); rr[m] = (C[pi[m]] < cv[m][:, None]).sum(1)
            m = np.flatnonzero((pj >= b) & (pj < b + 64)); rc[m] = (CT[pj[m]] < cv[m][:, None]).sum(1)
        del CT
        Cp = np.partition(C, 1, 1); rowbest, row2 = Cp[:, 0], Cp[:, 1]; del Cp; colbest = C.min(0)
        X = pair_feats(F, pi, pj, C, rr, rc, rowbest, row2, colbest, L)
        yv = (tl[pi] == pj)
        out[f"pi{L}"] = pi.astype(np.int32); out[f"pj{L}"] = pj.astype(np.int32); out[f"X{L}"] = X; out[f"y{L}"] = yv
        h = tl >= 0; recall.append(yv.sum() / h.sum())
    np.savez(os.path.join(OUT, "cache", f"feat_s{s}.npz"), **out)
    return f"sbj {s} n={n}: candidates/row {len(out['pi0']) / n:.1f}, true-successor recall " + " ".join(f"{r:.3f}" for r in recall) + f" [{time.time() - t0:.0f}s]"


if __name__ == "__main__":
    S = load_stage(); subs = [int(x) for x in a.subjects.split(",")] if a.subjects else [int(x) for x in np.unique(S["oof_sbj"])]
    subs = sorted(subs, key=lambda s: -(S["oof_sbj"] == s).sum())
    for m in Parallel(n_jobs=a.jobs)(delayed(run_subject)(S, s, a.K, a.Kc) for s in subs):
        print(m, flush=True)
