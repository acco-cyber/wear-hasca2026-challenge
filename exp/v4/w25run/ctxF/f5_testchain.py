"""step 4a: TEST chains on the 2025 rows (de-augmented, every twin replaced by our clean tile): chain LightGBM trained on all
22 training subjects' cache/feat_s*.npz (params / train mask / rounds of train_chain.py), candidates + features exactly as
chain_feats.run_subject (K=25, Kc=10, nll3), kernel lk_assign.  -> cache/test_chain_s{s}.npz (rows{L} = 2025 row ids of
pipeline limb L, succ{L}, conf{L} over positions in rows{L})"""
import os, sys, time, glob
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
sys.path.insert(0, FEAS)
_argv = sys.argv; sys.argv = [sys.argv[0]]
from chain_feats import tile_stats, pair_feats, FEATS          # chain_feats parses argv at import
sys.argv = _argv
from sim_chain import cost_matrix
sys.path.insert(0, os.path.join(W, "exp", "v4"))
from relink import lk_assign

P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, max_bin=63)
ROUNDS = 200; K, Kc = 25, 10
T0 = time.time()
MP = os.path.join(CACHE, "chain_full.txt")


def train():
    subs = sorted(int(f.split("_s")[-1][:-4]) for f in glob.glob(os.path.join(FEAS, "cache", "feat_s*.npz")))
    assert len(subs) == 22
    Xs, ys = [], []
    for s in subs:
        z = np.load(os.path.join(FEAS, "cache", f"feat_s{s}.npz"))
        for L in range(4):
            Xs.append(z[f"X{L}"]); ys.append(z[f"y{L}"])
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.int8); del Xs
    rr, rc = X[:, FEATS.index("rank_row")], X[:, FEATS.index("rank_col")]
    rng = np.random.default_rng(0); m = (y == 1) | (rr < 4) | (rc < 4) | (rng.random(len(y)) < 0.05)
    print(f"chain model: {len(y)} pairs, training on {m.sum()} [{time.time() - T0:.0f}s]", flush=True)
    mdl = lgb.train(P, lgb.Dataset(X[m], y[m]), ROUNDS); mdl.save_model(MP)
    print(f"chain model saved [{time.time() - T0:.0f}s]", flush=True)


def chain_one(s, L, Tl):
    t0 = time.time(); n = len(Tl)
    C = cost_matrix(Tl, "nll3"); F = tile_stats(Tl)
    top = np.argpartition(C, K, 1)[:, :K]; topc = np.argpartition(C, Kc, 0)[:Kc, :]
    pi = np.r_[np.repeat(np.arange(n), K), topc.ravel()]; pj = np.r_[top.ravel(), np.tile(np.arange(n), Kc)]
    key = np.unique(pi.astype(np.int64) * n + pj); pi, pj = key // n, key % n
    ok = pi != pj; pi, pj = pi[ok], pj[ok]
    cv = C[pi, pj]; rr = np.zeros(len(pi)); rc = np.zeros(len(pi)); CT = np.ascontiguousarray(C.T)
    for b in range(0, n, 64):
        m = np.flatnonzero((pi >= b) & (pi < b + 64)); rr[m] = (C[pi[m]] < cv[m][:, None]).sum(1)
        m = np.flatnonzero((pj >= b) & (pj < b + 64)); rc[m] = (CT[pj[m]] < cv[m][:, None]).sum(1)
    del CT
    Cp = np.partition(C, 1, 1); rowbest, row2 = Cp[:, 0], Cp[:, 1]; del Cp; colbest = C.min(0)
    X = pair_feats(F, pi, pj, C, rr, rc, rowbest, row2, colbest, L); del C
    mdl = lgb.Booster(model_file=MP); lo = mdl.predict(X, raw_score=True, num_threads=2)
    cnt = np.bincount(pi, minlength=n); Km = int(cnt.max()); order = np.argsort(pi, kind="stable")
    start = np.r_[0, np.cumsum(cnt)][:-1]; slot = np.arange(len(pi)) - start[pi[order]]
    cand = np.full((n, Km), -1, np.int64); Lm = np.full((n, Km), -50.0, np.float32)
    cand[pi[order], slot] = pj[order]; Lm[pi[order], slot] = lo[order]
    succ, sc = lk_assign(cand, Lm)
    conf = np.where(succ >= 0, 1 / (1 + np.exp(-sc)), 0.0).astype(np.float32)
    return s, L, succ, conf, f"sbj {s} limb {L}: n={n} linked {np.mean(succ >= 0):.3f} mean conf {conf.mean():.3f} q(conf) {np.quantile(conf, [0.1, 0.5, 0.9]).round(3)} [{time.time() - t0:.0f}s]"


if __name__ == "__main__":
    if not os.path.exists(MP):
        train()
    cz = np.load(os.path.join(CACHE, "w25_clean.npz")); acc, s25, l25 = cz["acc"], cz["sbj"], cz["limb"]
    jobs = []
    for s in (22, 23, 24, 25):
        for L in range(4):
            r = np.flatnonzero((s25 == s) & (l25 == PIPE_TO_25[L])); jobs.append((s, L, r))
    jobs.sort(key=lambda t: -len(t[2]))
    res = Parallel(n_jobs=3)(delayed(chain_one)(s, L, acc[r].astype(np.float64)) for s, L, r in jobs)
    out = {s: {} for s in (22, 23, 24, 25)}
    for (s, L, r), (s_, L_, succ, conf, msg) in zip(jobs, res):
        assert s == s_ and L == L_; print(msg, flush=True)
        out[s][f"rows{L}"] = r; out[s][f"succ{L}"] = succ; out[s][f"conf{L}"] = conf
    for s in out:
        np.savez(os.path.join(CACHE, f"test_chain_s{s}.npz"), **out[s])
    print(f"done [{time.time() - T0:.0f}s]")
