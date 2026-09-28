"""Local re-run of the Hanbat notebook's tabular experts + graph propagation + count calibration from the kept arrays
of our GPU fork (work/hanbat/keep), optionally with OUR base-model probabilities injected as extra stacker columns
and/or blended into the window log-probs.

python hanbat_stack.py test  [--extra v3b,v1,fusion] [--extra_to S3|both] [--win_blend v3b:0.3,...] [--tag name]
    -> work/hanbat/<tag>_test_logp_b.npy, <tag>_P_test.npy, subs/sub_hb_<tag>.csv
python hanbat_stack.py cv    [same options]   -> OOF macro-F1 of: window blend, tab blend, graph (L0 links proxy)
"""
import os, sys, json, re, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, scipy.sparse as sp
import lightgbm as lgb
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb"); HB = os.path.join(W, "work", "hanbat")
KEEP = os.path.join(HB, "keep")
N_CLS = 19
CFG = {"imu_w": 0.3, "lp": (10, 0.1, 0.5, 0.9), "link_alpha": 0.5, "link_iters": 10, "sharpen_T": 0.5, "per_ex": 97,
       "null_min": 0.05}
TRAIN_SETS = {0: 2, 14: 2}
TAB_PARAMS = dict(objective="multiclass", num_class=19, learning_rate=0.08, num_leaves=31, min_data_in_leaf=80,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=63,
                  num_threads=6, verbose=-1, seed=0)
TAB_ROUNDS = {"S3": 60, "T": 113}
TAB_TEST_CHOICE = (("S3", 0.5), ("T", 0.3))
NB_IMU = ["mean_x", "mean_y", "mean_z", "std_m", "energy_m", "mad_x", "mad_y", "mad_z", "jerk_m", "domf_m",
          "acmax_m", "aclag_m", "logpow_m", "grav_x", "grav_y", "grav_z"]
TEST_CFGS = ["g0.5/abs0.8/b-2.0/a0.95/r3w0.5", "g0.5/abs0.8/b-2.0/a0.93/r3w0.5", "g0.5/abs0.8/b-2.0/a0.93/r2w0.7",
             "g0.5/abs0.6/b-2.0/a0.93/r3w0.5", "g0.5/abs0.8/b-2.0/a0.95/r2w0.7"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1); present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def calibrate(P, sbj, sets, per_ex, null_min, iters=50):
    out = np.empty_like(P)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); n = len(ii)
        null = max(n - 18 * per_ex * sets.get(int(s), 1), null_min * n)
        t_ = np.full(P.shape[1], (n - null) / 18); t_[0] = null
        Q = P[ii].copy()
        for _ in range(iters):
            Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]; Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out


def finish(P, dd):
    Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"])
    Q = calibrate(Q / Q.sum(1, keepdims=True), dd["sbj"], dd["sets"], CFG["per_ex"], CFG["null_min"])
    return Q.argmax(1)


# ---------------- tabular experts ----------------
def neighbor_block(dd, imu, names):
    succ = dd["succ"]; n = len(succ)
    prv = np.full(n, -1); prv[succ[succ >= 0]] = np.flatnonzero(succ >= 0)
    sc_n = np.where(succ >= 0, dd["score"], np.nan); sc_p = np.where(prv >= 0, dd["score"][np.maximum(prv, 0)], np.nan)

    def take(A_, idx):
        out = A_[np.maximum(idx, 0)].astype(np.float32); out[idx < 0] = np.nan; return out
    nxt2 = np.where(succ >= 0, succ[np.maximum(succ, 0)], -1); prv2 = np.where(prv >= 0, prv[np.maximum(prv, 0)], -1)
    lp = dd["logp"]; P = np.exp(lp); acc = np.zeros_like(P); cnt = np.zeros(n)
    for idx in (prv2, prv, succ, nxt2):
        acc += np.where(idx[:, None] >= 0, P[np.maximum(idx, 0)], 0); cnt += idx >= 0
    ctx = np.log(np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), np.nan) + 1e-6)
    zi = [names.index("z_" + k) for k in NB_IMU]; sens = dd["sensor"].astype(np.float32)
    blocks = [take(lp, succ), take(lp, prv), ctx, sc_n[:, None], sc_p[:, None], cnt[:, None],
              take(sens[:, None], succ), take(sens[:, None], prv), take(imu[:, zi], succ), take(imu[:, zi], prv)]
    return np.concatenate(blocks, 1).astype(np.float32)


def knn_block(dd, k=10):
    E = dd["emb"] / (np.linalg.norm(dd["emb"], axis=1, keepdims=True) + 1e-6); P = np.exp(dd["logp"])
    out = np.zeros((len(P), 21), np.float32)
    for s in np.unique(dd["sbj"]):
        ii = np.flatnonzero(dd["sbj"] == s); S = E[ii] @ E[ii].T; np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]; sims = np.take_along_axis(S, nb, 1)
        out[ii, :19] = np.log(P[ii][nb].mean(1) + 1e-6); out[ii, 19], out[ii, 20] = sims.max(1), sims.mean(1)
    return out


def tab_build(variant, dd, feat, names, extra):
    X = [feat["imu"], feat["vmot"], feat["vpca"]]
    if variant in ("S1", "S2", "S3"):
        X += [dd["logp"], dd["fusion_logp"], dd["imu_logp"]]
    if variant in ("S2", "S3"):
        X.append(neighbor_block(dd, feat["imu"], names))
    if variant == "S3":
        X.append(knn_block(dd))
    if extra is not None:
        X.append(extra)
    return np.concatenate(X, 1).astype(np.float32)


def tab_blend(lp, parts):
    x = (1 - sum(w for _, w in parts)) * lp
    for q, w in parts:
        x = x + w * q
    return lsm(x).astype(np.float32)


# ---------------- graph propagation ----------------
def _norm_rows(P):
    P = np.clip(P, 1e-12, None); return P / P.sum(1, keepdims=True)


class SubjectKNN:
    def __init__(self, emb, grp, K=40):
        E = emb.astype(np.float32); self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        self.grp = grp; self.groups = [np.flatnonzero(grp == g) for g in np.unique(grp)]; self.K = K; self.cache = {}

    def topk(self, P0, use_p, key):
        ck = (key, use_p)
        if ck in self.cache:
            return self.cache[ck]
        out = []
        for ii in self.groups:
            S = self.E[ii] @ self.E[ii].T
            if use_p:
                Q = np.sqrt(P0[ii]).astype(np.float32); S += use_p * (Q @ Q.T)
            np.fill_diagonal(S, -np.inf)
            nb = np.argpartition(-S, self.K, axis=1)[:, :self.K]; vals = np.take_along_axis(S, nb, 1)
            o = np.argsort(-vals, 1); out.append((ii, np.take_along_axis(nb, o, 1), np.take_along_axis(vals, o, 1))); del S
        self.cache[ck] = out
        return out


def knn_W(nb, vals, k, tau):
    n = len(nb); nbk, vk = nb[:, :k], vals[:, :k]; rows = np.repeat(np.arange(n), k)
    w = np.exp((vk - vk.max()) / tau).reshape(-1)
    Wm = sp.csr_matrix((w, (rows, nbk.reshape(-1))), shape=(n, n)); return Wm.maximum(Wm.T).tocsr()


def row_norm(Wm):
    d_ = np.asarray(Wm.sum(1)).ravel(); d_[d_ == 0] = 1; return (sp.diags(1 / d_) @ Wm).tocsr()


def sym_norm(Wm):
    d_ = np.asarray(Wm.sum(1)).ravel(); d_[d_ == 0] = 1; Dm = sp.diags(1 / np.sqrt(d_)); return (Dm @ Wm @ Dm).tocsr()


def link_W(succ, score, n_total, T=1.0, b=0.0, power=1.0):
    m = succ >= 0; src, dst = np.flatnonzero(m), succ[m]
    wt = (1 / (1 + np.exp(-np.clip((score[m] - b) / T, -20, 20)))) ** power
    return sp.csr_matrix((np.r_[wt, wt], (np.r_[src, dst], np.r_[dst, src])), shape=(n_total, n_total))


def label_prop_generic(P0, knn, use_p=0.5, k=10, tau=0.1, alpha=0.9, iters=20, norm="sym", Qsrc=None, Lw=None,
                       beta=0.0, key="p0", link_mode="row"):
    tops = knn.topk(P0 if Qsrc is None else Qsrc, use_p, key); out = np.empty_like(P0)
    for ii, nb, vals in tops:
        Wm = knn_W(nb, vals, k, tau); A_ = sym_norm(Wm) if norm == "sym" else row_norm(Wm); f0 = P0[ii]
        if Lw is not None and beta > 0:
            Wl = Lw[ii][:, ii]; dl = np.asarray(Wl.sum(1)).ravel(); Al = row_norm(Wl)
            bb = beta * (dl > 0)[:, None] if link_mode == "row" else beta * np.minimum(dl / 2, 1)[:, None]
        f = f0.copy()
        for _ in range(iters):
            g = A_ @ f
            if Lw is not None and beta > 0:
                if norm == "sym":
                    gl = Al @ f; s_g = g.sum(1, keepdims=True) / np.maximum(f.sum(1, keepdims=True), 1e-12)
                    g = (1 - bb) * g + bb * gl * s_g
                else:
                    g = (1 - bb) * g + bb * (Al @ f)
            f = alpha * g + (1 - alpha) * f0
        out[ii] = f
    return _norm_rows(out)


def smooth_emb(emb, Lw, g=0.5, h=1):
    E = emb.astype(np.float32); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
    if g == 0:
        return E
    An = row_norm(Lw); Es = E.copy()
    for _ in range(h):
        Es = E + g * (An @ Es)
    return Es.astype(np.float32)


class Runner:
    def __init__(self, dd, K=40, b=0.0):
        self.d = dd; self.P0 = np.exp(dd["logp"]); self.Lw = link_W(dd["succ"], dd["score"], len(self.P0), b=b)
        self.knns, self.K, self.stage1 = {}, K, {}

    def knn(self, emb_g):
        if emb_g not in self.knns:
            E = smooth_emb(self.d["emb"], self.Lw, emb_g, 1) if emb_g else self.d["emb"]
            self.knns[emb_g] = SubjectKNN(E, self.d["grp"], self.K)
        return self.knns[emb_g]

    def cal(self, P, T):
        return calibrate(_norm_rows(P ** T), self.d["sbj"], self.d["sets"], CFG["per_ex"], CFG["null_min"])

    def __call__(self, cfg):
        c = dict(alpha=0.95, beta=0.5, tau=0.1, k=10, iters=20, use_p=0.5, link_mode="row", emb_g=0.0, rounds=1, T=2.0,
                 w=0.5, alpha2=None); c.update(cfg)
        knn = self.knn(c["emb_g"])
        kw = dict(use_p=c["use_p"], k=c["k"], tau=c["tau"], iters=c["iters"], norm="row", Lw=self.Lw, beta=c["beta"],
                  link_mode=c["link_mode"], key="p0")
        k1 = (c["emb_g"], c["alpha"], repr(tuple(sorted(kw.items(), key=lambda x: x[0]))))
        if k1 not in self.stage1:
            self.stage1[k1] = label_prop_generic(self.P0, knn, alpha=c["alpha"], **kw)
        F = self.stage1[k1]; a2 = c["alpha2"] if c["alpha2"] is not None else c["alpha"]
        for _ in range(c["rounds"] - 1):
            Q = self.cal(F, c["T"])
            F = label_prop_generic(_norm_rows(self.P0 ** (1 - c["w"]) * Q ** c["w"]), knn, Qsrc=self.P0, alpha=a2, **kw)
        return F


def parse_cfg(name):
    m = re.fullmatch(r"g([\d.]+)/(abs|row)([\d.]+)/b(-?[\d.]+)/a([\d.]+)/r(\d)w([\d.]+)", name)
    g, lm, beta, b, a, r, w = m.groups()
    return dict(emb_g=float(g), link_mode=lm, beta=float(beta), link_b=float(b), alpha=float(a), rounds=int(r), T=2.0, w=float(w))


def graph_P(dd, cfg_names):
    Ps, runners = [], {}
    for c in sorted((parse_cfg(nm) for nm in cfg_names), key=lambda c: c["link_b"]):
        b = c.pop("link_b")
        if b not in runners:
            runners.clear(); runners[b] = Runner(dd, b=b)
        Ps.append(runners[b](c).astype(np.float32))
    return _norm_rows(np.mean(Ps, 0))


# ---------------- data ----------------
def load_all():
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    bl = np.load(os.path.join(KEEP, "blend.npz")); l0 = np.load(os.path.join(KEEP, "links_L0.npz"))
    names = json.load(open(os.path.join(KEEP, "imu_names.json")))
    fo = dict(np.load(os.path.join(KEEP, "feat_oof.npz"))); ft = dict(np.load(os.path.join(KEEP, "feat_test.npz")))
    d = {"logp": bl["oof_logp"], "fusion_logp": bl["oof_fusion"], "imu_logp": bl["oof_imu"],
         "emb": np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32), "grp": sm["sbj"], "y": sm["y"], "sbj": sm["sbj"],
         "fold": sm["fold"], "sensor": sm["sensor"], "sets": TRAIN_SETS, "succ": l0["oof_succ"].astype(np.int64),
         "score": l0["oof_score"].astype(np.float32)}
    l2 = np.load(os.path.join(KEEP, "links_L2_test.npz"))
    t = {"logp": bl["test_logp"], "fusion_logp": bl["test_fusion"], "imu_logp": bl["test_imu"],
         "emb": np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32), "grp": bl["test_sbj"].astype(np.int64),
         "sbj": bl["test_sbj"].astype(np.int64), "sensor": bl["test_sensor"].astype(np.int64), "sets": {},
         "succ": l2["succ"].astype(np.int64), "score": l2["score_qn"].astype(np.float32),
         "l0_succ": l0["test_succ"].astype(np.int64), "l0_score": l0["test_score"].astype(np.float32)}
    return d, t, fo, ft, names


def load_extra(names_):
    if not names_:
        return None, None
    O, T = [], []
    for nm in names_.split(","):
        o = np.load(os.path.join(HYB, f"base_{nm}_oof.npy")); tt = np.load(os.path.join(HYB, f"base_{nm}_test.npy"))
        O.append(np.log(np.clip(o, 1e-6, 1)).astype(np.float32)); T.append(np.log(np.clip(tt, 1e-6, 1)).astype(np.float32))
    return np.concatenate(O, 1), np.concatenate(T, 1)


def p0_mix(LB, spec, split):
    """log-softmax(LB + sum w_i * log base_i): base_i = base_<name>_{split}.npy, or 'path=<file>' (test only)"""
    if not spec:
        return LB
    x = LB.astype(np.float64).copy()
    for part in spec.split(","):
        nm, w = part.rsplit(":", 1)
        if nm.startswith("path="):
            if split != "test":
                continue
            p = nm[5:]
        else:
            p = os.path.join(HYB, f"base_{nm}_{split}.npy")
            if not os.path.exists(p):
                log(f"p0 mix: {p} missing, skipped"); continue
        b = np.load(p).astype(np.float64); b = b / b.sum(1, keepdims=True)
        x += float(w) * np.log(np.clip(b, 1e-6, 1))
    return lsm(x).astype(np.float32)


def apply_win_blend(dd, spec, split):
    """replace dd['logp'] by log-softmax(logp + sum w_i * log base_i)"""
    if not spec:
        return dd
    x = dd["logp"].astype(np.float64).copy()
    for part in spec.split(","):
        nm, w = part.split(":")
        b = np.load(os.path.join(HYB, f"base_{nm}_{split}.npy")); x += float(w) * np.log(np.clip(b, 1e-6, 1))
    dd = dict(dd); dd["logp"] = lsm(x).astype(np.float32); return dd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["test", "cv"])
    ap.add_argument("--extra", default=""); ap.add_argument("--extra_to", default="S3", choices=["S3", "both"])
    ap.add_argument("--win_blend", default=""); ap.add_argument("--tag", default="base")
    ap.add_argument("--s3w", type=float, default=0.5); ap.add_argument("--tw", type=float, default=0.3)
    ap.add_argument("--graph_links", default="L2", choices=["L2", "L0"])
    ap.add_argument("--p0", default="", help="log-mix external probabilities into the graph input, e.g. v3b:0.5,path=work/probs_x.npy:1.0")
    a = ap.parse_args()
    d, t, fo, ft, names = load_all()
    d = apply_win_blend(d, a.win_blend, "oof"); t = apply_win_blend(t, a.win_blend, "test")
    xo, xt = load_extra(a.extra)
    log(f"window-level OOF macro F1 {macro_f1(d['y'], d['logp'].argmax(1)):.4f}")
    parts = (("S3", a.s3w), ("T", a.tw))
    if a.mode == "cv":
        oof_tab = {}
        for v, _ in parts:
            ex = xo if (v == "S3" or a.extra_to == "both") else None
            cache = os.path.join(HB, "cache_T_oof.npy") if (v == "T" and ex is None and not a.win_blend) else None
            if cache and os.path.exists(cache):
                oof_tab[v] = np.load(cache); log(f"{v}: OOF from cache, macro F1 {macro_f1(d['y'], oof_tab[v].argmax(1)):.4f}"); continue
            X = tab_build(v, d, fo, names, ex); Q = np.zeros((len(X), N_CLS), np.float32)
            for f in range(5):
                tr, va = d["fold"] != f, d["fold"] == f
                bst = lgb.train(TAB_PARAMS, lgb.Dataset(X[tr], d["y"][tr], params={"max_bin": 63}), TAB_ROUNDS[v])
                Q[va] = np.log(np.clip(bst.predict(X[va]), 1e-7, 1))
            oof_tab[v] = Q; log(f"{v}: OOF macro F1 {macro_f1(d['y'], Q.argmax(1)):.4f} (X {X.shape})"); del X
            if cache:
                np.save(cache, Q)
        LB = tab_blend(d["logp"], [(oof_tab[v], w) for v, w in parts])
        log(f"tab blend OOF macro F1 {macro_f1(d['y'], LB.argmax(1)):.4f}")
        LB = p0_mix(LB, a.p0, "oof")
        if a.p0:
            log(f"tab blend + p0 mix OOF macro F1 {macro_f1(d['y'], LB.argmax(1)):.4f}")
        np.save(os.path.join(HB, f"{a.tag}_oof_logp_b.npy"), LB)
        dg = dict(logp=LB, emb=d["emb"], grp=d["sbj"], sbj=d["sbj"], succ=d["succ"], score=d["score"], sets=TRAIN_SETS)
        P = graph_P(dg, TEST_CFGS); lab = finish(P, dg)
        log(f"graph (L0 links proxy) OOF macro F1 {macro_f1(d['y'], lab):.4f}   per fold "
            + " ".join(f"{f}:{macro_f1(d['y'][d['fold'] == f], lab[d['fold'] == f]):.4f}" for f in range(5)))
        np.save(os.path.join(HB, f"{a.tag}_oof_P.npy"), P)
        return
    TAB = {}
    for v, _ in parts:
        ex = xo if (v == "S3" or a.extra_to == "both") else None; ext = xt if ex is not None else None
        X = tab_build(v, d, fo, names, ex)
        bst = lgb.train(TAB_PARAMS, lgb.Dataset(X, d["y"], params={"max_bin": 63}), TAB_ROUNDS[v]); del X
        Xt = tab_build(v, t, ft, names, ext); TAB[v] = np.log(np.clip(bst.predict(Xt), 1e-7, 1)).astype(np.float32)
        log(f"{v}: test argmax distribution {np.bincount(TAB[v].argmax(1), minlength=N_CLS).tolist()}")
    LB = tab_blend(t["logp"], [(TAB[v], w) for v, w in parts])
    LB = p0_mix(LB, a.p0, "test")
    np.save(os.path.join(HB, f"{a.tag}_test_logp_b.npy"), LB)
    succ, score = (t["succ"], t["score"]) if a.graph_links == "L2" else (t["l0_succ"], t["l0_score"])
    dt = dict(logp=LB, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], succ=succ, score=score, sets={})
    P = graph_P(dt, TEST_CFGS); np.save(os.path.join(HB, f"{a.tag}_P_test.npy"), P)
    lab = finish(P, dt)
    ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True); assert (ids == np.arange(len(ids))).all()
    out = os.path.join(W, "subs", f"sub_hb_{a.tag}.csv")
    pd.DataFrame({"id": ids, "target_feature": lab.astype(int)}).to_csv(out, index=False)
    ref = np.load(os.path.join(KEEP, "blend.npz"))["labels"]
    log(f"wrote {out}; counts {np.bincount(lab, minlength=N_CLS).tolist()}; agreement with kernel labels {np.mean(lab == ref):.4f}")


if __name__ == "__main__":
    main()
