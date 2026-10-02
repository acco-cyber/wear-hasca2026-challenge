"""Per-window refiner features from graph_<mode>.npz (gen_graph.py).  python feats.py cv|test -> feats_<mode>.npz"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP, N_CLS
W = r"E:\Claude code\wear"; OUT = os.path.dirname(os.path.abspath(__file__))


def inv(succ):
    pred = np.full(len(succ), -1, np.int64); m = succ >= 0; pred[succ[m]] = np.flatnonzero(m); return pred


def hop(a, b):
    return np.where(a >= 0, b[np.maximum(a, 0)], -1)


def nb_share(nbs, lab, cand):
    cnt = np.zeros(len(cand)); hit = np.zeros(len(cand))
    for nb in nbs:
        ok = nb >= 0; cnt += ok; hit += ok & (lab[np.maximum(nb, 0)] == cand)
    return np.where(cnt > 0, hit / np.maximum(cnt, 1), np.nan), cnt


def rank_of(M, c):
    v = M[np.arange(len(c)), c]; return (M > v[:, None]).sum(1).astype(np.float32)


def pct_within(x, grp, mask=None):
    """within-group percentile of x; with mask, ranked among the masked rows only (others get NaN) so the value does not
    depend on how many rows of the subject disagree"""
    out = np.full(len(x), np.nan, np.float32)
    for g in np.unique(grp):
        ii = np.flatnonzero((grp == g) if mask is None else ((grp == g) & mask))
        if len(ii) == 0:
            continue
        r = np.argsort(np.argsort(x[ii], kind="stable"), kind="stable"); out[ii] = r / max(len(ii) - 1, 1)
    return out


def build(mode):
    g = np.load(os.path.join(OUT, f"graph_{mode}.npz"))
    P, Q, t, o, has, sbj = g["P"].astype(np.float64), g["Q"].astype(np.float64), g["base"], g["ours"], g["has"], g["sbj"]
    n = len(t); ar = np.arange(n)
    bl = np.load(os.path.join(KEEP, "blend.npz"))
    if mode == "cv":
        lp0 = np.load(os.path.join(W, "work", "hanbat", "cv_base_oof_logp_b.npy")).astype(np.float64); lw = bl["oof_logp"].astype(np.float64)
        emb = np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32)
    else:
        lp0 = np.load(os.path.join(KEEP, "test_logp_b.npy")).astype(np.float64); lw = bl["test_logp"].astype(np.float64)
        emb = np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32)
    F = {}
    Qs = -np.sort(-Q, 1)
    F["q1"], F["q2"], F["q3"] = Qs[:, 0], Qs[:, 1], Qs[:, 2]
    F["margin"] = Qs[:, 0] - Qs[:, 1]
    F["entQ"] = -(Q * np.log(np.clip(Q, 1e-12, 1))).sum(1)
    F["Q_o"] = Q[ar, o]; F["Q_t"] = Q[ar, t]
    F["lrQ"] = np.log(np.clip(Q[ar, o], 1e-12, 1)) - np.log(np.clip(Q[ar, t], 1e-12, 1))
    F["rkQ_o"] = rank_of(Q, o)
    F["q1_pct"] = pct_within(Qs[:, 0], sbj); F["margin_pct"] = pct_within(F["margin"], sbj)
    dis = t != o
    F["lrQ_pct"] = pct_within(F["lrQ"], sbj, dis)
    F["rkP_o"] = rank_of(P, o); F["rkP_t"] = rank_of(P, t)
    F["lrP"] = np.log(np.clip(P[ar, o], 1e-12, 1)) - np.log(np.clip(P[ar, t], 1e-12, 1))
    for nm, M in (("P0", lp0), ("W", lw)):
        F[f"rk{nm}_o"] = rank_of(M, o); F[f"rk{nm}_t"] = rank_of(M, t)
        F[f"lr{nm}"] = M[ar, o] - M[ar, t]
        F[f"lr{nm}_pct"] = pct_within(F[f"lr{nm}"], sbj, dis)
    sp_ = "oof" if mode == "cv" else "test"
    for nm, key in (("F", f"{sp_}_fusion"), ("I", f"{sp_}_imu")):     # single-modality window evidence
        M = bl[key].astype(np.float64); F[f"lr{nm}"] = M[ar, o] - M[ar, t]; F[f"rk{nm}_o"] = rank_of(M, o)
    # link neighbours: their links (L0 on cv / L2 on test) and our chain links
    for nm, su in (("L", g["succ"]), ("C", g["s2"])):
        pr = inv(su); nbs = [hop(pr, pr), pr, su, hop(su, su)]
        for ln, lab in (("b", t), ("u", o)):
            for cn, c in (("t", t), ("o", o)):
                F[f"{nm}_{ln}{cn}"], cnt = nb_share(nbs, lab, c)
        F[f"{nm}_n"] = cnt
        s1, _ = nb_share([pr, su], t, o); F[f"{nm}1_bo"] = s1       # 1-hop: their label on direct neighbours == ours
        acc = np.zeros(n); c_ = np.zeros(n)                          # neighbours' window evidence for o vs t
        for nb in nbs:
            ok = nb >= 0; j = np.maximum(nb, 0); acc += np.where(ok, lw[j, o] - lw[j, t], 0); c_ += ok
        F[f"{nm}_lrW"] = np.where(c_ > 0, acc / np.maximum(c_, 1), np.nan)
    # video kNN within subject (k=10)
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6); k = 10
    kb = {c: np.zeros(n, np.float32) for c in ("bt", "bo", "ut", "uo")}; ksim = np.zeros(n, np.float32)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); S = E[ii] @ E[ii].T; np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]; ksim[ii] = np.take_along_axis(S, nb, 1).mean(1)
        tb, ob = t[ii][nb], o[ii][nb]
        kb["bt"][ii] = (tb == t[ii][:, None]).mean(1); kb["bo"][ii] = (tb == o[ii][:, None]).mean(1)
        kb["ut"][ii] = (ob == t[ii][:, None]).mean(1); kb["uo"][ii] = (ob == o[ii][:, None]).mean(1); del S
        L = lw[ii]; kb.setdefault("lrW", np.zeros(n, np.float32))[ii] = (L[nb, o[ii][:, None]] - L[nb, t[ii][:, None]]).mean(1)
    for c, v in kb.items():
        F[f"K_{c}"] = v
    F["K_sim"] = ksim
    # class ids + subject-level stats
    F["cls_t"] = t.astype(np.float32); F["cls_o"] = o.astype(np.float32)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); m = len(ii)
        cb = np.bincount(t[ii], minlength=N_CLS) / m; co = np.bincount(o[ii], minlength=N_CLS) / m
        for key, val in (("S_n", np.full(m, m)), ("S_null_b", np.full(m, cb[0])), ("S_null_u", np.full(m, co[0])),
                         ("S_dis", np.full(m, np.mean(t[ii] != o[ii]))), ("S_cb_o", cb[o[ii]]), ("S_cu_o", co[o[ii]]),
                         ("S_cb_t", cb[t[ii]]), ("S_cu_t", co[t[ii]])):
            F.setdefault(key, np.zeros(n, np.float32))[ii] = val
    names = list(F.keys()); X = np.stack([np.asarray(F[k], np.float32) for k in names], 1)
    extra = dict(t=t, o=o, has=has, sbj=sbj, gate_lab=g["gate_lab"])
    if mode == "cv":
        sm = np.load(os.path.join(KEEP, "sim_meta.npz")); extra.update(y=sm["y"].astype(np.int64), fold=sm["fold"].astype(np.int64))
    np.savez_compressed(os.path.join(OUT, f"feats_{mode}.npz"), X=X, names=np.array(names), **extra)
    print(mode, X.shape, "disagree rate (has rows)", round(float(np.mean(t[has] != o[has])), 4))


if __name__ == "__main__":
    t0 = time.time(); build(sys.argv[1]); print(f"{time.time() - t0:.0f}s")
