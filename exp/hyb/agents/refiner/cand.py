"""Candidate refiner (task 2): per row, candidates = top-3 of their calibrated Q + our label; pointwise LightGBM
'candidate is the true class'; pick argmax if its score beats their label's by > delta.
python cand.py [--test out.csv] [--rounds 600] [--nocls]"""
import os, sys, argparse
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import KEEP, N_CLS, macro_f1
from feats import inv, hop, rank_of, pct_within
W = r"E:\Claude code\wear"; OUT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--test", default=""); ap.add_argument("--rounds", type=int, default=600)
ap.add_argument("--nocls", action="store_true"); ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()


def load(mode):
    g = np.load(os.path.join(OUT, f"graph_{mode}.npz")); bl = np.load(os.path.join(KEEP, "blend.npz"))
    sp_ = "oof" if mode == "cv" else "test"
    lp0 = np.load(os.path.join(W, "work", "hanbat", "cv_base_oof_logp_b.npy") if mode == "cv" else os.path.join(KEEP, "test_logp_b.npy")).astype(np.float64)
    emb = np.load(os.path.join(KEEP, f"{sp_}_emb.npy")).astype(np.float32)
    return g, dict(P0=lp0, W=bl[f"{sp_}_logp"].astype(np.float64), F=bl[f"{sp_}_fusion"].astype(np.float64), I=bl[f"{sp_}_imu"].astype(np.float64)), emb


def build(mode):
    g, M, emb = load(mode)
    P, Q, t, o, has, sbj = g["P"].astype(np.float64), g["Q"].astype(np.float64), g["base"], g["ours"], g["has"], g["sbj"]
    n = len(t); ar = np.arange(n); srt = np.argsort(-Q, 1)
    cands = [srt[:, 0], srt[:, 1], srt[:, 2], np.where((o == srt[:, 0]) | (o == srt[:, 1]) | (o == srt[:, 2]), -1, o)]
    # shared neighbour structures
    nbl = {}
    for nm, su in (("L", g["succ"]), ("C", g["s2"])):
        pr = inv(su); nbl[nm] = [hop(pr, pr), pr, su, hop(su, su)]
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6); knn = np.zeros((n, 10), np.int64)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); S_ = E[ii] @ E[ii].T; np.fill_diagonal(S_, -np.inf)
        knn[ii] = ii[np.argpartition(-S_, 10, axis=1)[:, :10]]; del S_
    shares = {}
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        shares[int(s)] = (np.bincount(t[ii], minlength=N_CLS) / len(ii), np.bincount(o[ii], minlength=N_CLS) / len(ii))
    cb = np.stack([shares[int(s)][0] for s in sbj]); cu = np.stack([shares[int(s)][1] for s in sbj])
    Qs = -np.sort(-Q, 1); row = dict(q1_pct=pct_within(Qs[:, 0], sbj), margin_pct=pct_within(Qs[:, 0] - Qs[:, 1], sbj),
                                      agree=(o == t).astype(np.float32), null_b=cb[:, 0], null_u=cu[:, 0], cls_t=t.astype(np.float32))
    lw = M["W"]; blocks, rid, slot, cc = [], [], [], []
    for k, c in enumerate(cands):
        ok = c >= 0; c0 = np.maximum(c, 0); F = {}
        F["slot"] = np.full(n, k, np.float32); F["cls_c"] = c0.astype(np.float32); F["is_o"] = (c0 == o).astype(np.float32)
        F["rkQ"] = rank_of(Q, c0); F["rkP"] = rank_of(P, c0)
        for nm, A in M.items():
            F[f"rk{nm}"] = rank_of(A, c0); F[f"lr{nm}"] = A[ar, c0] - A[ar, t]
        for nm, nbs in nbl.items():
            cnt = np.zeros(n); hb = np.zeros(n); hu = np.zeros(n); ev = np.zeros(n)
            for nb in nbs:
                v = nb >= 0; j = np.maximum(nb, 0); cnt += v; hb += v & (t[j] == c0); hu += v & (o[j] == c0)
                ev += np.where(v, lw[j, c0] - lw[j, t], 0)
            d = np.maximum(cnt, 1); F[f"{nm}_b"] = np.where(cnt > 0, hb / d, np.nan); F[f"{nm}_u"] = np.where(cnt > 0, hu / d, np.nan)
            F[f"{nm}_lrW"] = np.where(cnt > 0, ev / d, np.nan)
        F["K_b"] = (t[knn] == c0[:, None]).mean(1); F["K_u"] = (o[knn] == c0[:, None]).mean(1)
        F["K_lrW"] = (lw[knn, c0[:, None]] - lw[knn, t[:, None]]).mean(1)
        F["S_cb"] = cb[ar, c0]; F["S_cu"] = cu[ar, c0]
        F.update(row)
        names = list(F.keys()); X = np.stack([np.asarray(F[k_], np.float32) for k_ in names], 1)
        blocks.append(X[ok]); rid.append(ar[ok]); slot.append(np.full(ok.sum(), k)); cc.append(c0[ok])
    return dict(X=np.concatenate(blocks), rid=np.concatenate(rid), slot=np.concatenate(slot), c=np.concatenate(cc), names=names,
                t=t, o=o, has=has, sbj=sbj, n=n)


cv = build("cv"); te = build("test")
sm = np.load(os.path.join(KEEP, "sim_meta.npz")); y = sm["y"].astype(np.int64); fold = sm["fold"].astype(np.int64)
g = np.load(os.path.join(OUT, "graph_cv.npz")); gl = g["gate_lab"]
names = cv["names"]; drop = {"cls_c", "cls_t"} if a.nocls else set(); keep = [i for i, nm in enumerate(names) if nm not in drop]
fn = [names[i] for i in keep]; cat = [fn.index(c) for c in ("cls_c", "cls_t") if c in fn]
X, rid, cc = cv["X"][:, keep], cv["rid"], cv["c"]; S = cv["has"]
use = S[rid]; X, rid, cc = X[use], rid[use], cc[use]; tgt = (cc == y[rid]).astype(np.float32); cfold = fold[rid]
print(f"candidate rows {len(X)} for {S.sum()} rows; oracle (true class among candidates) {np.mean(np.isin(np.flatnonzero(S), rid[tgt == 1])):.4f}; "
      f"feats {len(fn)}", flush=True)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, num_threads=4, verbose=-1, max_bin=63, cat_smooth=20, min_data_per_group=100)
sc = np.zeros(len(X)); imp = np.zeros(len(fn))
for f in range(5):
    tr, va = cfold != f, cfold == f
    for sd in range(a.seeds):
        bst = lgb.train(dict(params, seed=sd), lgb.Dataset(X[tr], tgt[tr], categorical_feature=cat), a.rounds)
        sc[va] += bst.predict(X[va]) / a.seeds; imp += bst.feature_importance("gain")


def decide(n, rid, cc, sc, t, delta):
    best = np.full(n, -1.0); arg = t.copy(); st = np.full(n, np.nan)
    m_t = cc == t[rid]; st[rid[m_t]] = sc[m_t]
    o_ = np.argsort(sc, kind="stable")                     # ascending -> later writes win = highest score
    best[rid[o_]] = sc[o_]; arg[rid[o_]] = cc[o_]
    lab = t.copy(); sw = (arg != t) & (best - np.nan_to_num(st, nan=0.0) > delta); lab[sw] = arg[sw]; return lab


t = cv["t"]; grid = np.round(np.arange(-0.05, 0.301, 0.01), 2)
f1s = [macro_f1(y[S], decide(cv["n"], rid, cc, sc, t, d)[S]) for d in grid]; dbest = grid[int(np.argmax(f1s))]
print("delta curve: " + " ".join(f"{d:.2f}:{v:.4f}" for d, v in zip(grid[::3], f1s[::3])))
lab = decide(cv["n"], rid, cc, sc, t, dbest); lab0 = decide(cv["n"], rid, cc, sc, t, 0.0)
lab_n = t.copy()
for f in range(5):
    m = S & (fold != f); fs_ = [macro_f1(y[m], decide(cv["n"], rid, cc, sc, t, d)[m]) for d in grid]; d_f = grid[int(np.argmax(fs_))]
    lf = decide(cv["n"], rid, cc, sc, t, d_f); mm = fold == f; lab_n[mm] = lf[mm]
for nm, L in (("their labels", t), ("hand gate 0.55 top2", gl), (f"cand refiner delta={dbest}", lab), ("cand refiner nested", lab_n), ("cand refiner delta=0", lab0)):
    print(f"{nm:30s} {macro_f1(y[S], L[S]):.4f}  " + " ".join(f"{macro_f1(y[S & (fold == f)], L[S & (fold == f)]):.4f}" for f in range(5))
          + f"  changed {np.mean(L[S] != t[S]):.4f}")
o_ = np.argsort(-imp); print("importance: " + ", ".join(f"{fn[i]}={imp[i] / imp.sum():.3f}" for i in o_[:12]))
np.savez(os.path.join(OUT, "oof_cand.npz"), lab=lab, lab_n=lab_n, dbest=dbest)
if a.test:
    Xt = te["X"][:, keep]; st_ = np.zeros(len(Xt))
    for sd in range(a.seeds):
        bst = lgb.train(dict(params, seed=sd), lgb.Dataset(X, tgt, categorical_feature=cat), a.rounds); st_ += bst.predict(Xt) / a.seeds
    np.savez(os.path.join(OUT, "cand_scores.npz"), sc=sc, rid=rid, cc=cc, st=st_, trid=te["rid"], tc=te["c"], dbest=dbest)
    labt = decide(te["n"], te["rid"], te["c"], st_, te["t"], dbest)
    ref = pd.read_csv(os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv")).sort_values("id").target_feature.to_numpy()
    print(f"TEST changed {np.mean(labt != te['t']):.4f} (OOF {np.mean(lab[S] != t[S]):.4f}); to ours {np.mean((labt != te['t']) & (labt == te['o'])):.4f}; "
          f"agreement with gl7 best {np.mean(labt == ref):.4f}")
    pd.DataFrame({"id": np.arange(te["n"]), "target_feature": labt.astype(int)}).to_csv(os.path.join(OUT, a.test), index=False)
