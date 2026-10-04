"""Link-level fusion of independent fits: average the L3 link-scorer log-odds of every candidate pair over the fits
(union of the fits' candidate lists; a pair missing from a fit's list gets that row's weakest log-odds minus 1), then
the kernel's Hungarian assignment with cycle breaking, 7 Gumbel-perturbed matchings (tau 0.3) and the per-subject
quantile normalisation of the scores. Writes a links.npz in the keep4 format (oof_/test_ succ, score).
  python relink.py --fits K1,K2[,...] --out links_fused.npz [--members 8] [--tau 0.3]"""
import os, sys, argparse, time
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment


def lk_assign(cl, lo, floor=-50.0):
    """kernel code: Hungarian 1:1 -> drop invalid links -> break every cycle at its weakest link"""
    n = len(cl)
    cost = np.full((n, n), floor, np.float32)
    valid = cl >= 0
    rows = np.broadcast_to(np.arange(n)[:, None], cl.shape)
    cost[rows[valid], cl[valid]] = lo[valid]
    np.fill_diagonal(cost, -1e4)
    r, c = linear_sum_assignment(-cost)
    succ = np.full(n, -1, np.int64); succ[r] = c
    edge = cost[np.arange(n), succ]; succ[edge <= floor + 1] = -1
    seen = np.zeros(n, bool)
    for s in range(n):
        if seen[s] or succ[s] < 0:
            continue
        path, pos, node = [], {}, s
        while node >= 0 and not seen[node]:
            seen[node] = True; pos[node] = len(path); path.append(node); node = int(succ[node])
        if node >= 0 and node in pos:
            cyc = path[pos[node]:]
            succ[cyc[int(np.argmin([cost[x, succ[x]] for x in cyc]))]] = -1
    score = np.where(succ >= 0, cost[np.arange(n), np.maximum(succ, 0)], -50.0).astype(np.float32)
    return succ, score


def perturbed(cand, L, tau, seed):
    g = np.random.default_rng(seed).gumbel(size=L.shape).astype(np.float32)
    s_loc, _ = lk_assign(cand, np.where(cand >= 0, L + tau * g, L))
    hit = cand == s_loc[:, None]
    return s_loc, np.where(s_loc >= 0, np.where(hit, L, -np.inf).max(1), -50.0).astype(np.float32)


def qnorm(succ, score, sbj, ref):
    ref = np.sort(np.asarray(ref, np.float64)); out = score.astype(np.float32).copy()
    for s in np.unique(sbj):
        m = np.flatnonzero((sbj == s) & (succ >= 0))
        if not len(m):
            continue
        r = np.argsort(np.argsort(score[m], kind="stable"), kind="stable"); out[m] = np.quantile(ref, (r + 0.5) / len(m)).astype(np.float32)
    return out


def fuse_subject(cands, Ls, w):
    """union of candidate lists, weighted mean log-odds; returns (cand (n,K), L (n,K))"""
    n = cands[0].shape[0]; F = len(cands)
    keys, vals, which = [], [], []
    fill = []
    for f, (c, L) in enumerate(zip(cands, Ls)):
        v = (c >= 0) & (L > -49)
        rows = np.broadcast_to(np.arange(n)[:, None], c.shape)
        keys.append(rows[v].astype(np.int64) * n + c[v]); vals.append(L[v].astype(np.float64)); which.append(np.full(v.sum(), f))
        rmin = np.where(v, L, np.inf).min(1); rmin[~np.isfinite(rmin)] = -10.0; fill.append(rmin - 1.0)
    K_ = np.concatenate(keys); Vv = np.concatenate(vals); Wh = np.concatenate(which)
    uk, inv = np.unique(K_, return_inverse=True)
    acc = np.zeros(len(uk)); have = np.zeros((len(uk), F), bool)
    np.add.at(acc, inv, Vv * w[Wh]); have[inv, Wh] = True
    rows_u = uk // n
    for f in range(F):                                   # pairs this fit did not list: its row's weakest score - 1
        miss = ~have[:, f]; acc[miss] += w[f] * fill[f][rows_u[miss]]
    cols_u = uk % n
    cnt = np.bincount(rows_u, minlength=n); K = int(cnt.max())
    cand = np.full((n, K), -1, np.int64); Lm = np.full((n, K), -50.0, np.float32)
    start = np.r_[0, np.cumsum(cnt)][:-1]; pos = np.arange(len(uk)) - start[rows_u]
    cand[rows_u, pos] = cols_u; Lm[rows_u, pos] = acc
    return cand, Lm


def build(split, Z, sbj, w, members, tau, ref, jobs, true_succ=None):
    subs = [int(s) for s in np.unique(sbj)]
    fused = Parallel(n_jobs=jobs)(delayed(fuse_subject)([z[f"{split}_{s}_cand"] for z in Z], [z[f"{split}_{s}_L"] for z in Z], w) for s in subs)
    jobs_ = [(k, i) for k in range(members) for i in range(len(subs))]
    res = Parallel(n_jobs=jobs)(delayed(lk_assign if k == 0 else perturbed)(*((fused[i][0], fused[i][1]) if k == 0 else (fused[i][0], fused[i][1], tau, 1000 * k + subs[i])))
                                for k, i in jobs_)
    S, C = np.full((members, len(sbj)), -1, np.int64), np.full((members, len(sbj)), -50.0, np.float32)
    for (k, i), (s_loc, sc) in zip(jobs_, res):
        ii = np.flatnonzero(sbj == subs[i]); S[k, ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1); C[k, ii] = sc
    for k in range(members):
        C[k] = qnorm(S[k], C[k], sbj, ref)
    if true_succ is not None:
        m = S[0] >= 0
        print(f"{split}: fused plain matching linked {m.mean():.3f}, exact successor {(S[0][m] == true_succ[m]).mean():.4f}; "
              f"perturbed mean exact {np.mean([(S[k][S[k] >= 0] == true_succ[S[k] >= 0]).mean() for k in range(1, members)]):.4f}", flush=True)
    return S, C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--w", default=""); ap.add_argument("--members", type=int, default=8); ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(); t0 = time.time()
    dirs = a.fits.split(","); Z = [np.load(os.path.join(d, "link_logodds.npz")) for d in dirs]
    st = np.load(os.path.join(dirs[0], "stage.npz"), allow_pickle=True); lk = np.load(os.path.join(dirs[0], "links.npz"))
    w = np.array([float(x) for x in a.w.split(",")]) if a.w else np.ones(len(dirs)); w = w / w.sum()
    sbj, tsbj, ts = st["oof_sbj"].astype(np.int64), st["test_sbj"].astype(np.int64), st["true_succ"].astype(np.int64)
    for d in dirs:                                       # each fit's own plain matching, for reference
        l_ = np.load(os.path.join(d, "links.npz")); s0 = l_["oof_succ"][0]; m = s0 >= 0
        print(f"{os.path.basename(os.path.dirname(d.rstrip(chr(92) + '/')))}: own plain matching exact successor {(s0[m] == ts[m]).mean():.4f}", flush=True)
    So, Co = build("oof", Z, sbj, w, a.members, a.tau, lk["qn_ref"], a.jobs, ts)
    St, Ct = build("test", Z, tsbj, w, a.members, a.tau, lk["qn_ref"], a.jobs)
    np.savez(a.out, oof_succ=So, oof_score=Co, test_succ=St, test_score=Ct)
    print(f"wrote {a.out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
