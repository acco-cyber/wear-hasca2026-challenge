"""Transductive per-subject experts, cross-fitted in two halves on each subject's own confident fused labels.
Halves are blocks of 15 consecutive tiles along the first K7 matching's chains (so a tile's own bout neighbourhood is
mostly in its own half and its pseudo-label never trains its own prediction). Same recipe for training and test
subjects; nothing uses oof_y.
  python selfx.py [--conf 0.3] [--dims 64] [--block 15]  -> cache/self.npz"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1

HERE = os.path.dirname(os.path.abspath(__file__))
W = r"E:\Claude code\wear"
K7 = W + r"\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = W + r"\work\v4\wear-v4-big-tf-opt-s9\keep4"
NC = 19
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def chain_blocks(succ, ii, block, rng):
    """half id (0/1) for every tile of one subject from blocks along the matching's chains"""
    n_all = len(succ); loc = -np.ones(n_all, np.int64); loc[ii] = np.arange(len(ii))
    su = np.where(succ[ii] >= 0, loc[np.maximum(succ[ii], 0)], -1)
    su[(succ[ii] >= 0) & (loc[np.maximum(succ[ii], 0)] < 0)] = -1        # never leave the subject
    n = len(ii); pr = np.full(n, -1); ok = su >= 0; pr[su[ok]] = np.flatnonzero(ok)
    half = np.full(n, -1); seen = np.zeros(n, bool)
    starts = list(np.flatnonzero(pr < 0)) + list(range(n))
    for s0 in starts:
        if seen[s0]:
            continue
        cur, pos = s0, 0; h = rng.integers(2)
        while cur >= 0 and not seen[cur]:
            seen[cur] = True
            if pos % block == 0 and pos > 0:
                h = rng.integers(2)
            half[cur] = h; cur = su[cur]; pos += 1
    return half


def pca_feats(E, dims):
    X = E.astype(np.float64) - E.mean(0)
    ev, U = np.linalg.eigh(np.cov(X.T)); o = np.argsort(-ev)[:dims]
    Z = X @ U[:, o]; return Z / (Z.std(0) + 1e-6)


def fit_predict(kind, Xtr, ytr, Xte):
    cls = np.unique(ytr); out = np.full((len(Xte), NC), -20.0)
    if len(cls) < 2:
        out[:, cls] = 0; return out
    if kind == "lda":
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    else:
        m = LogisticRegression(C=0.1, max_iter=300).fit(Xtr, ytr)
    out[:, m.classes_] = np.log(np.clip(m.predict_proba(Xte), 1e-9, None))
    return lsm(out)


def run(split, Q, lab, sbj, emb, succ, B2, args, rng):
    n = len(lab); conf = Q.max(1)
    res = {k: np.zeros((n, NC)) for k in ("lda", "lrb")}
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        cr = np.argsort(np.argsort(conf[ii])) / len(ii)
        half = chain_blocks(succ, ii, args.block, rng)
        Z = pca_feats(emb[ii], args.dims)
        Zb = np.concatenate([Z, B2[ii]], 1)
        for h in (0, 1):
            tr = (half != h) & (cr >= args.conf); te = half == h
            res["lda"][ii[te]] = fit_predict("lda", Z[tr], lab[ii[tr]], Z[te])
            res["lrb"][ii[te]] = fit_predict("lr", Zb[tr], lab[ii[tr]], Zb[te])
    log(f"{split}: done")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conf", type=float, default=0.3); ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--block", type=int, default=15); ap.add_argument("--out", default="self")
    a = ap.parse_args()
    z = np.load(os.path.join(HERE, "cache", "feat.npz"))
    y, sbj, tsbj, fold = z["y"], z["sbj"], z["tsbj"], z["fold"]
    Qo, Qt, lo, lt = z["Qo"], z["Qt"], z["lo"], z["lt"]
    st7 = np.load(os.path.join(K7, "stage.npz")); st9 = np.load(os.path.join(K9, "stage.npz"))
    B2o = lsm(0.5 * st7["B2_OOF"].astype(np.float64) + 0.5 * st9["B2_OOF"]); B2t = lsm(0.5 * st7["B2_TEST"].astype(np.float64) + 0.5 * st9["B2_TEST"])
    lk = np.load(os.path.join(K7, "links.npz"))
    eo = np.load(os.path.join(K7, "oof_emb.npy")).astype(np.float32); et = np.load(os.path.join(K7, "test_emb.npy")).astype(np.float32)
    rng = np.random.default_rng(0)
    ro = run("o", Qo, lo, sbj, eo, lk["oof_succ"][0].astype(np.int64), B2o, a, rng)
    rt = run("t", Qt, lt, tsbj, et, lk["test_succ"][0].astype(np.int64), B2t, a, rng)
    cr = np.zeros(len(y))
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); cr[ii] = np.argsort(np.argsort(Qo[ii].max(1))) / len(ii)
    low = cr <= 0.1
    for k in ro:
        p = ro[k].argmax(1)
        log(f"{k}: alone OOF F1 {macro_f1(y, p):.4f}, agree with fused {np.mean(p == lo):.4f}, test agree {np.mean(rt[k].argmax(1) == lt):.4f}; "
            f"bottom-10% tiles: acc {np.mean(p[low] == y[low]):.4f} vs fused {np.mean(lo[low] == y[low]):.4f}; "
            f"where it disagrees with fused: it is right {np.mean(p[p != lo] == y[p != lo]):.3f}, fused right {np.mean(lo[p != lo] == y[p != lo]):.3f}")
    out = {}
    for k in ro:
        out[f"{k}_o"] = ro[k].astype(np.float32); out[f"{k}_t"] = rt[k].astype(np.float32)
        out[f"{k}r_o"] = (ro[k] - ro[k].max(1, keepdims=True)).astype(np.float32); out[f"{k}r_t"] = (rt[k] - rt[k].max(1, keepdims=True)).astype(np.float32)
    np.savez(os.path.join(HERE, "cache", f"{a.out}.npz"), **out)
    log("saved")


if __name__ == "__main__":
    main()
