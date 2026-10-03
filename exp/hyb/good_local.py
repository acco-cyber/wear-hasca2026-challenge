"""Local re-implementation of the final decode stage of the public "Learned Links + Counts" notebook, from the stage
inputs kept by our fork (work/good/keep3): bagged link matchings -> graph decode per matching (geometric mean) ->
learned count prior (ridge on margin profiles, nested by fold) in two passes -> Sinkhorn with count targets.
Extra independent link sets can be added as edges in every decode (--xl) on the OOF rows (our sim chain links, 18
sessions) and on the test rows (work/test_structure.pkl), to validate the union in CV before using it on test.
  python good_local.py [--xl 1.0] [--tag name] [--members 8] [--jobs 4] [--test_only]"""
import os, sys, argparse, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hanbat_stack as H
from hanbat_stack import macro_f1, KEEP, TRAIN_SETS, TEST_CFGS, N_CLS, CFG, _norm_rows, parse_cfg
from graph_lab import Runner2, calibrate_targets, default_targets
W = r"E:\Claude code\wear"; GK = os.path.join(W, "work", "good", "keep3"); HYB = os.path.join(W, "exp", "hyb")
RK = [60, 70, 80, 90, 97, 105, 115, 125, 140]


def graph_P_one(dd, targets):
    Ps, runners = [], {}
    for c in sorted((parse_cfg(nm) for nm in TEST_CFGS), key=lambda c: c["link_b"]):
        b = c.pop("link_b")
        if b not in runners:
            runners.clear(); runners[b] = Runner2(dd, targets, None, 0.0, b=b)
        Ps.append(runners[b](c).astype(np.float32))
    return _norm_rows(np.mean(Ps, 0))


def bag_P(base, links, targets, jobs):
    """links: list of (succ, score); geometric mean of the per-matching graph decodes"""
    def one(su, sc):
        return graph_P_one(dict(base, succ=su, score=sc), targets)
    Ps = Parallel(n_jobs=jobs)(delayed(one)(su, sc) for su, sc in links)
    G = np.exp(np.mean([np.log(np.clip(q, 1e-9, None)) for q in Ps], 0)); return G / G.sum(1, keepdims=True)


def profile_features(mats, sbj, sets):
    X, key = [], []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); ns = sets.get(int(s), 1)
        for c in range(1, N_CLS):
            f = []
            for M in mats:
                q = M[ii][:, c]; sc = np.sort(np.log(q + 1e-6) - np.log(np.delete(M[ii], c, 1).max(1) + 1e-6))[::-1]
                f += [sc[min(int(r * ns), len(sc) - 1)] for r in RK]
                f += [q.sum() / ns, (M[ii].argmax(1) == c).sum() / ns, (sc > 0).sum() / ns, (sc > -1).sum() / ns, (sc > 1).sum() / ns]
            X.append(f + [len(ii) / ns]); key.append((int(s), c))
    return np.array(X, np.float64), np.array(key)


def ridge_cv(X, y, Xte, alphas=(1, 3, 10, 30, 100, 300, 1000)):
    """RidgeCV with leave-one-out error (same selection rule as sklearn's default), intercept fitted"""
    xm, ym = X.mean(0), y.mean(); Xc, yc = X - xm, y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); Uy = U.T @ yc; best = None
    for a in alphas:
        d = s ** 2 / (s ** 2 + a); pred = U @ (d * Uy); h = (U ** 2) @ d + 1.0 / len(y)
        loo = np.mean(((yc - pred) / (1 - h)) ** 2)
        if best is None or loo < best[0]:
            best = (loo, a)
    a = best[1]; coef = Vt.T @ ((s / (s ** 2 + a)) * Uy)
    return (Xte - xm) @ coef + ym


def fit_counts(Xtr, ttr, Xte):
    reg = (ttr > 55) & (ttr < 150); mu, sd = Xtr[reg].mean(0), Xtr[reg].std(0) + 1e-6
    return np.clip(ridge_cv((Xtr[reg] - mu) / sd, ttr[reg], (Xte - mu) / sd), 70, 135)


def count_targets(sbj, sets, key, cnt):
    out = {}
    for s in np.unique(sbj):
        n = int((sbj == s).sum()); per = cnt[key[:, 0] == s] * sets.get(int(s), 1)
        null = max(n - per.sum(), CFG["null_min"] * n); out[int(s)] = np.r_[null, per * (n - null) / per.sum()]
    return out


def finish_targets(P, sbj, targets):
    Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); return calibrate_targets(Q / Q.sum(1, keepdims=True), sbj, targets)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xl", type=float, default=0.0); ap.add_argument("--tag", default="base")
    ap.add_argument("--members", type=int, default=8); ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args(); t0 = time.time()
    st = np.load(os.path.join(GK, "stage.npz")); lk = np.load(os.path.join(GK, "links.npz"))
    y, sbj, fold, tsbj = st["oof_y"].astype(np.int64), st["oof_sbj"].astype(np.int64), st["oof_fold"].astype(np.int64), st["test_sbj"].astype(np.int64)
    fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
    dd = dict(logp=st["B2_OOF"].astype(np.float32), emb=np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32), grp=sbj, sbj=sbj, sets=TRAIN_SETS)
    dt = dict(logp=st["B2_TEST"].astype(np.float32), emb=np.load(os.path.join(GK, "test_emb.npy")).astype(np.float32), grp=tsbj, sbj=tsbj, sets={})
    Lo = [(lk["oof_succ"][k].astype(np.int64), lk["oof_score"][k].astype(np.float32)) for k in range(min(a.members, len(lk["oof_succ"])))]
    Lt = [(lk["test_succ"][k].astype(np.int64), lk["test_score"][k].astype(np.float32)) for k in range(min(a.members, len(lk["test_succ"])))]
    if a.xl > 0:
        sys.path.insert(0, os.path.join(W, "exp", "transductive")); from tlib import load_structs
        o2t = np.load(os.path.join(HYB, "rows.npz"))["ours_to_theirs"]; s2 = np.full(len(y), -1, np.int64); c2 = np.full(len(y), -50.0, np.float32)
        for which in ("eval", "extra", "extra2"):
            for s, d_ in load_structs(which).items():
                a0, n = int(d_["a"]), int(d_["n"]); su = np.asarray(d_["succ0"]); sc = np.asarray(d_["sc"], np.float32); ok = (su >= 0) & (sc >= -6.0)
                rows_t = o2t[a0 + np.arange(n)]; s2[rows_t[ok]] = o2t[a0 + su[ok]]; c2[rows_t[ok]] = sc[ok]
        dd.update(succ2=s2, score2=c2, xl_w=a.xl, xl_b=-2.0)
        struct = pickle.load(open(os.path.join(W, "work", "test_structure.pkl"), "rb")); t2 = np.full(len(tsbj), -1, np.int64); tc2 = np.full(len(tsbj), -50.0, np.float32)
        for s, d_ in struct.items():
            idx = np.asarray(d_["idx"]); su = np.asarray(d_["succ0"]); sc = np.asarray(d_["sc"], np.float32); ok = (su >= 0) & (sc >= -6.0)
            t2[idx[ok]] = idx[su[ok]]; tc2[idx[ok]] = sc[ok]
        dt.update(succ2=t2, score2=tc2, xl_w=a.xl, xl_b=-2.0)
        print(f"extra links: OOF coverage {np.mean(s2 >= 0):.3f}, test coverage {np.mean(t2 >= 0):.3f}", flush=True)
    Bp, Btp = np.exp(dd["logp"].astype(np.float64)), np.exp(dt["logp"].astype(np.float64))
    tg0, tgt0 = default_targets(sbj, TRAIN_SETS), default_targets(tsbj, {})
    Pt = bag_P(dt, Lt, tgt0, a.jobs); print(f"test pass 0 done [{time.time() - t0:.0f}s]", flush=True)
    if a.test_only:      # count regressor from the kernel's cached OOF profile features
        dc = np.load(os.path.join(GK, "dec_cache.npz"))
        for k in range(a.passes):
            X, true = dc[f"stage_B_{k}_X"], dc[f"stage_B_{k}_true"]
            Xt, kt = profile_features([Pt, Btp], tsbj, {}); ct = fit_counts(X, true, Xt); tgt = count_targets(tsbj, {}, kt, ct)
            if k < a.passes - 1:
                Pt = bag_P(dt, Lt, tgt, a.jobs)
    else:
        P = bag_P(dd, Lo, tg0, a.jobs); print(f"oof pass 0 done [{time.time() - t0:.0f}s]; bagged, fixed 97: F1 {macro_f1(y, finish_targets(P, sbj, tg0).argmax(1)):.4f}", flush=True)
        for k in range(a.passes):
            X, key = profile_features([P, Bp], sbj, TRAIN_SETS)
            true = np.array([(y[sbj == s] == c).sum() / TRAIN_SETS.get(int(s), 1) for s, c in key], np.float64); kf = np.array([fold_of[int(s)] for s, _ in key])
            cnt = np.zeros(len(true))
            for f in range(5):
                cnt[kf == f] = fit_counts(X[kf != f], true[kf != f], X[kf == f])
            tg = count_targets(sbj, TRAIN_SETS, key, cnt)
            Xt, kt = profile_features([Pt, Btp], tsbj, {}); ct = fit_counts(X, true, Xt); tgt = count_targets(tsbj, {}, kt, ct)
            lab = finish_targets(P, sbj, tg).argmax(1)
            print(f"pass {k + 1}: count error {np.abs(cnt - true).mean():.2f}; OOF F1 {macro_f1(y, lab):.4f} per fold "
                  + " ".join(f"{macro_f1(y[fold == f], lab[fold == f]):.4f}" for f in range(5)) + f" [{time.time() - t0:.0f}s]", flush=True)
            if k < a.passes - 1:
                P, Pt = bag_P(dd, Lo, tg, a.jobs), bag_P(dt, Lt, tgt, a.jobs)
    Qt = finish_targets(Pt, tsbj, tgt)
    out = os.path.join(W, "subs", f"sub_goodlocal_{a.tag}.csv")
    pd.DataFrame({"id": np.arange(len(tsbj)), "target_feature": Qt.argmax(1).astype(int)}).to_csv(out, index=False)
    np.save(out.replace(".csv", "_Q.npy"), Qt.astype(np.float32))
    ref = st["QB_TEST"].argmax(1); print(f"wrote {out}; agreement with the kernel's final labels {np.mean(Qt.argmax(1) == ref):.4f} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
