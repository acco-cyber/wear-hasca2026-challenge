"""Definitive validation of chain reconstruction using TRAIN data with known order.
Simulates the test construction for one train subject: 1-s windows tiled with stride 1 s, video frames 30t+8..30t+22,
inertial samples 50t..50t+49 from ONE random limb; shuffles; rebuilds chains with video-only and video+inertial LLR
+ linear assignment; reports exact edge precision (succ == t+1) and label purity of chains.
Usage: python 05_train_chain_sim.py sbj_0 [sbj_5 ...]   (needs train/inertial_feat/<s>.csv and train/videomae_feat/<s>.npy)"""
import numpy as np, pandas as pd, os, sys, time
from scipy.optimize import linear_sum_assignment
D = r"E:\Claude code\wear\data\train"; A = r"E:\Claude code\wear\research\artifacts"
LIMBS = ["right_arm", "right_leg", "left_leg", "left_arm"]
rng = np.random.default_rng(0)

def llr(pos, neg, x, bins=60):
    lo = min(pos.min(), neg.min()); hi = max(np.percentile(pos, 99.9), np.percentile(neg, 99.9)); e = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(pos, e); hn, _ = np.histogram(neg, e); hp = (hp + 1) / (hp.sum() + bins); hn = (hn + 1) / (hn.sum() + bins)
    k = np.clip(np.searchsorted(e, x, side="right") - 1, 0, bins - 1); return np.log(hp[k]) - np.log(hn[k])

def lsa_chain(L):
    n = len(L); r, c = linear_sum_assignment(-L); succ = c.copy(); sc = L[r, c]; succ[sc < np.percentile(sc, 3)] = -1
    pred = np.full(n, -1); pred[succ[succ >= 0]] = np.where(succ >= 0)[0]
    # cut weakest edge in each cycle
    seen = np.zeros(n, bool)
    for st in range(n):
        if seen[st]: continue
        cur = st; k = 0
        while pred[cur] >= 0 and k <= n:
            cur = pred[cur]; k += 1
            if cur == st: break
        if pred[cur] >= 0:
            cyc = [cur]; x = succ[cur]
            while x != cur: cyc.append(x); x = succ[x]
            w = np.array([L[a, succ[a]] for a in cyc]); a = cyc[int(w.argmin())]; pred[succ[a]] = -1; succ[a] = -1
        x = cur
        while x >= 0 and not seen[x]: seen[x] = True; x = succ[x]
    return succ

for s in sys.argv[1:] or ["sbj_0"]:
    t0 = time.time()
    df = pd.read_csv(os.path.join(D, "inertial_feat", f"{s}.csv")); V = np.load(os.path.join(D, "videomae_feat", f"{s}.npy"), mmap_mode="r")
    lab = df["label"].fillna("null").astype(str).values
    acc = df[[f"{l}_acc_{a}" for l in LIMBS for a in "xyz"]].values.reshape(-1, 4, 3)
    T = min(len(df) // 50, V.shape[0] // 30)
    limb = rng.integers(0, 4, T)
    Xi = np.stack([acc[50 * t: 50 * t + 50, limb[t]] for t in range(T)])                       # (T,50,3)
    Vw = np.stack([np.asarray(V[30 * t + 8: 30 * t + 23], dtype=np.float32) for t in range(T)])  # (T,15,768)
    y = np.array([pd.Series(lab[50 * t: 50 * t + 50]).mode()[0] for t in range(T)])
    print(f"{s}: T={T} windows, video frames {V.shape[0]} (={V.shape[0]/30:.1f}s) inertial {len(df)/50:.1f}s; null frac {np.mean(y=='null'):.3f}")
    Vn = Vw / (np.linalg.norm(Vw, axis=2, keepdims=True) + 1e-9)
    tail = Vn[:, 12:15].mean(1); head = Vn[:, 0:3].mean(1); tail /= np.linalg.norm(tail, axis=1, keepdims=True); head /= np.linalg.norm(head, axis=1, keepdims=True)
    Sv = tail @ head.T; np.fill_diagonal(Sv, -1)
    true_adj = Sv[np.arange(T - 1), np.arange(1, T)]
    ra = rng.choice(T, 40000); rb = rng.choice(T, 40000); m = (ra != rb) & (rb != ra + 1)
    print(f"  TRUE adjacent tail3/head3 cos: median {np.median(true_adj):.3f} p10 {np.percentile(true_adj,10):.3f} | random median {np.median(Sv[ra[m], rb[m]]):.3f}")
    within = (Vn[:, 0:3].mean(1) * Vn[:, 12:15].mean(1)).sum(1) / (np.linalg.norm(Vn[:, 0:3].mean(1), axis=1) * np.linalg.norm(Vn[:, 12:15].mean(1), axis=1))
    print(f"  within-window proxy cos median {np.median(within):.3f} (proxy used on test)")
    # top-1 successor accuracy (video only)
    succ1 = Sv.argmax(1); print(f"  video-only argmax successor == t+1: {np.mean(succ1[:-1] == np.arange(1, T)):.3f}; within +-2: {np.mean(np.abs(succ1[:-1] - np.arange(1, T)) <= 2):.3f}")
    Lv = llr(within, Sv[ra[m], rb[m]], Sv.ravel()).reshape(T, T); np.fill_diagonal(Lv, -1e6)
    G = np.linalg.norm(Xi[:, -1, None, :] - Xi[None, :, 0, :], axis=2); same = limb[:, None] == limb[None, :]
    pos_i = np.linalg.norm(np.diff(Xi, axis=1), axis=2).ravel(); neg_i = rng.choice(G[same].ravel(), 200000, replace=False)
    Li = np.zeros((T, T), np.float32); Li[same] = llr(pos_i, neg_i, G[same]); np.fill_diagonal(Li, 0)
    for name, L in (("video_only", Lv), ("video+inertial", Lv + Li)):
        succ = lsa_chain(L); e = np.where(succ >= 0)[0]
        exact = np.mean(succ[e] == e + 1); near = np.mean(np.abs(succ[e] - (e + 1)) <= 2)
        samelab = np.mean(y[succ[e]] == y[e])
        # chains
        pred = np.full(T, -1); pred[succ[e]] = e; starts = np.where(pred < 0)[0]; Lc = []
        for st in starts:
            k = 1; x = st
            while succ[x] >= 0: x = succ[x]; k += 1
            Lc.append(k)
        Lc = np.array(Lc)
        print(f"  [{name}] edges {len(e)}: EXACT succ precision {exact:.3f}; within +-2 s {near:.3f}; same-label {samelab:.3f} | chains {len(Lc)} max {Lc.max()} median {np.median(Lc):.0f}; nodes in chains>=100: {Lc[Lc>=100].sum()/T:.2f}")
    print("  %.0fs" % (time.time() - t0))
