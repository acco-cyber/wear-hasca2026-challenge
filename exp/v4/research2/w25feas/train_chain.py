"""(a, learned) nested per-limb continuity scorer: LightGBM on the chain_feats candidates, trained on the other four
subject folds, then per (subject, limb) Hungarian on log-odds with cycle breaking.  -> cache/chain_lgb_s{s}.npz
(succ{L}, conf{L} = predicted probability of the chosen link, rows, true_loc) + summary."""
import os, sys, time, glob
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
from common import *
sys.path.insert(0, os.path.join(W, "exp", "v4"))
from relink import lk_assign
from chain_feats import FEATS

P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, max_bin=63)
ROUNDS = 200
T0 = time.time()
S = load_stage(); fold_of = {int(s): int(f) for s, f in zip(S["oof_sbj"], S["oof_fold"])}
subs = sorted(int(f.split("_s")[-1][:-4]) for f in glob.glob(os.path.join(OUT, "cache", "feat_s*.npz")))
Z = {s: np.load(os.path.join(OUT, "cache", f"feat_s{s}.npz")) for s in subs}
Xs, ys, ss, Ls = [], [], [], []
for s in subs:
    for L in range(4):
        X = Z[s][f"X{L}"]; Xs.append(X); ys.append(Z[s][f"y{L}"]); ss.append(np.full(len(X), s)); Ls.append(np.full(len(X), L))
X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.int8); sb = np.concatenate(ss); fo = np.array([fold_of[int(s)] for s in sb])
del Xs
print(f"{len(X)} candidate pairs, {y.mean():.4f} positive [{time.time() - T0:.0f}s]", flush=True)
rr, rc = X[:, FEATS.index("rank_row")], X[:, FEATS.index("rank_col")]
rng = np.random.default_rng(0); train_mask = (y == 1) | (rr < 4) | (rc < 4) | (rng.random(len(y)) < 0.05)
pred = np.zeros(len(y), np.float32)
for f in range(5):
    tr = (fo != f) & train_mask; te = fo == f
    pf = os.path.join(OUT, "cache", f"chain_lgb_pred_f{f}.npy")
    if os.path.exists(pf):
        pred[te] = np.load(pf); print(f"fold {f}: cached", flush=True); continue
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr]), ROUNDS)
    pred[te] = m.predict(X[te], raw_score=True); np.save(pf, pred[te])
    if f == 0:
        imp = m.feature_importance("gain"); print("top features:", [FEATS[k] for k in np.argsort(-imp)[:10]], flush=True)
    print(f"fold {f}: trained on {tr.sum()} rows [{time.time() - T0:.0f}s]", flush=True)
np.save(os.path.join(OUT, "cache", "chain_lgb_pred.npy"), pred)


def assign_subject(s, preds, z):
    n = len(z["rows"]); out = dict(rows=z["rows"], true_loc=z["true_loc"]); tl = z["true_loc"]; h = tl >= 0; msg = []
    for L in range(4):
        pi, pj, lo = z[f"pi{L}"].astype(np.int64), z[f"pj{L}"].astype(np.int64), preds[L]
        cnt = np.bincount(pi, minlength=n); K = int(cnt.max()); order = np.argsort(pi, kind="stable")
        start = np.r_[0, np.cumsum(cnt)][:-1]; slot = np.arange(len(pi)) - start[pi[order]]
        cand = np.full((n, K), -1, np.int64); Lm = np.full((n, K), -50.0, np.float32)
        cand[pi[order], slot] = pj[order]; Lm[pi[order], slot] = lo[order]
        succ, sc = lk_assign(cand, Lm)
        out[f"succ{L}"] = succ; out[f"conf{L}"] = np.where(succ >= 0, 1 / (1 + np.exp(-sc)), 0.0).astype(np.float32)
        msg.append(f"L{L} {np.mean(succ[h] == tl[h]):.3f}")
    np.savez(os.path.join(OUT, "cache", f"chain_lgb_s{s}.npz"), **out)
    return f"sbj {s}: " + " ".join(msg)


off = 0; per = {}
for s in subs:
    per[s] = []
    for L in range(4):
        k = len(Z[s][f"y{L}"]); per[s].append(pred[off:off + k]); off += k
def zdict(s):
    z = Z[s]; return {k: z[k] for k in ["rows", "true_loc"] + [f"{p}{L}" for p in ("pi", "pj") for L in range(4)]}


del X
for m in Parallel(n_jobs=3)(delayed(assign_subject)(s, per[s], zdict(s)) for s in subs):
    print(m, flush=True)
print(f"done [{time.time() - T0:.0f}s]")
