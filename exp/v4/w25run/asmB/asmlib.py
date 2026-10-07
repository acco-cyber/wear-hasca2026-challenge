"""asmB: global offset solver for timeline assembly (fragments of per-limb 2025 chains aligned by link votes).
Data loaders for the permuted-node training simulation and the real test, the solver and the link derivation.
Nothing outside E:\\Claude code\\wear\\exp\\v4\\w25run\\asmB is written by this module."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

W = r"E:\Claude code\wear"
TD = os.path.join(W, "exp", "v4", "w25run", "testD")
CACHE = os.path.join(TD, "cache")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
HERE = os.path.dirname(os.path.abspath(__file__))

_STAGE = {}


def stage(fit=K7):
    if fit not in _STAGE:
        st = np.load(os.path.join(fit, "stage.npz"), allow_pickle=True)
        _STAGE[fit] = {k: st[k].astype(np.int64) for k in ["oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "true_succ",
                                                          "sensor_oof", "test_sbj", "sensor_test"]}
    return _STAGE[fit]


# ------------------------------------------------------------------------------------------------ loaders
def load_sim(s, seed=12345):
    """training subject s in the de-augmented test-like simulation with PERMUTED node ids per limb chain.
    Returns the solver inputs (lim, anch, pos, owner, succ, conf) and a separate 'truth' dict used only for scoring."""
    S = stage(); sbj = S["oof_sbj"]; N = len(sbj)
    loc = np.flatnonzero(sbj == s); n = len(loc)
    z = np.load(os.path.join(CACHE, f"chaindeaug_s{s}.npz"))
    rows = z["rows"]; assert len(rows) == n
    p_of = np.full(N, -1); p_of[rows] = np.arange(n)
    l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    tpos = p_of[loc]                                   # true position (time order) of our tile i -- scoring only
    lim = S["sensor_oof"][loc]
    anch = ~z["aug"][lim, tpos]
    rng = np.random.default_rng(seed + 1000 * s)
    perm = [rng.permutation(n) for _ in range(4)]      # old node (position) -> new node id
    succ, conf, tnode = [], [], []
    for L in range(4):
        su0 = z[f"succ{L}"].astype(np.int64); cf0 = z[f"conf{L}"].astype(np.float32)
        P = perm[L]; su = np.full(n, -1, np.int64); cf = np.zeros(n, np.float32)
        su[P] = np.where(su0 >= 0, P[np.maximum(su0, 0)], -1); cf[P] = cf0
        succ.append(su); conf.append(cf)
        inv = np.empty(n, np.int64); inv[P] = np.arange(n); tnode.append(inv)      # new node -> true position (scoring)
    pos = np.array([perm[lim[i]][tpos[i]] for i in range(n)], np.int64)
    pos = np.where(anch, pos, 0)
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    ts = S["true_succ"][loc]; tl = np.where(ts >= 0, l_of[np.maximum(ts, 0)], -1)
    # true next position of each position (for chain-link accuracy)
    tsr = S["true_succ"][rows]; tnp = np.where(tsr >= 0, p_of[np.maximum(tsr, 0)], -1)
    truth = dict(tpos=tpos, tl=tl, y=S["oof_y"][loc], tnode=tnode, tnp=tnp, rec=S["oof_rec"][rows])
    return dict(s=s, n=n, loc=loc, lim=lim, anch=anch, pos=pos, owner=owner, succ=succ, conf=conf,
                fold=int(S["oof_fold"][loc[0]]), truth=truth)


def load_test(s, fit=K7):
    S = stage(fit); tsbj = S["test_sbj"]
    cf = np.load(os.path.join(CACHE, f"testdeaug_chain_s{s}.npz"))
    loc = np.flatnonzero(tsbj == s); n = len(loc)
    assert (cf["loc"] == loc).all() and (cf["lim"] == S["sensor_test"][loc]).all()
    lim = cf["lim"].astype(np.int64); anch = cf["anch"].astype(bool); pos = np.where(anch, cf["pos"].astype(np.int64), 0)
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    assert ((cf["owner"] >= 0) == (owner >= 0)).all() and (cf["owner"][owner >= 0] == owner[owner >= 0]).all()
    succ = [cf[f"succ{L}"].astype(np.int64) for L in range(4)]; conf = [cf[f"conf{L}"].astype(np.float32) for L in range(4)]
    return dict(s=s, n=n, loc=loc, lim=lim, anch=anch, pos=pos, owner=owner, succ=succ, conf=conf, fold=-1, truth=None)


def local_links(Su, Sc, loc, NG):
    """global-index matchings (members, NG) -> subject-local (members, n)"""
    l_of = np.full(NG, -1); l_of[loc] = np.arange(len(loc))
    su = Su[:, loc]; su = np.where(su >= 0, l_of[np.maximum(su, 0)], -1)
    return su, Sc[:, loc].astype(np.float32)


# ------------------------------------------------------------------------------------------------ fragments
def fragments(succ, conf, thr):
    """per limb: confident chain links (conf >= thr) -> paths. Returns frag (4, n) global fragment id of each node,
    off (4, n) offset inside it, flim (F,), flen (F,), fnodes list of node arrays."""
    n = len(succ[0]); frag = np.full((4, n), -1, np.int64); off = np.zeros((4, n), np.int64)
    flim, fnodes = [], []
    for L in range(4):
        su = np.where(conf[L] >= thr, succ[L], -1)
        has_pred = np.zeros(n, bool); has_pred[su[su >= 0]] = True
        heads = np.flatnonzero(~has_pred)
        seen = np.zeros(n, bool)
        for h in heads:
            path = []; q = h
            while q >= 0 and not seen[q]:
                seen[q] = True; path.append(q); q = su[q]
            fid = len(fnodes); path = np.array(path, np.int64)
            frag[L, path] = fid; off[L, path] = np.arange(len(path)); flim.append(L); fnodes.append(path)
        for q in np.flatnonzero(~seen):              # residual cycles (should not happen): break arbitrarily
            if seen[q]:
                continue
            path = []; x = q
            while x >= 0 and not seen[x]:
                seen[x] = True; path.append(x); x = su[x]
            fid = len(fnodes); path = np.array(path, np.int64)
            frag[L, path] = fid; off[L, path] = np.arange(len(path)); flim.append(L); fnodes.append(path)
    return frag, off, np.array(flim), np.array([len(p) for p in fnodes]), fnodes
