"""probe 13: right_arm: global fixed-parameter types b = a @ M_k + N_k (M_k 3x3, N_k 50x3), alternating pair matching
and least-squares refits; reports residuals of true pairs once converged"""
import os, csv, numpy as np
np.set_printoptions(linewidth=250, suppress=True, precision=4)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
claimed = np.zeros(len(A25), bool); claimed[twin[ex]] = True
L = 2
bad = np.flatnonzero(~ex & (l26 == L))
p12 = np.load(os.path.join(D, "p12_rlin.npz")); r1, r2, arg = p12["r1"], p12["r2"], p12["arg"]
pz = np.load(os.path.join(D, "p10_rpat.npz")); pats = [np.zeros((50, 3)), pz["A"], pz["B"]]
dyn = np.linalg.norm(A26 - A26.mean(1, keepdims=True), axis=(1, 2))
kb = r1.argmin(1); rb = r1[np.arange(len(bad)), kb]; r2b = r2[np.arange(len(bad)), kb]
conf = (rb < 0.2 * r2b) & (dyn[bad] > np.median(dyn[bad]))


def fit(ai, bi):
    """min sum ||b - a M - N||^2 over M (3x3), N (50x3)"""
    am = ai.mean(0); bm = bi.mean(0)
    X = (ai - am).reshape(-1, 3); Y = (bi - bm).reshape(-1, 3)
    M = np.linalg.lstsq(X, Y, rcond=None)[0]; N = bm - am @ M
    return M, N


# init: cluster the confident pairs' per-pair M into types by nearest of two seeds
Ms = []
for q in np.flatnonzero(conf):
    k = kb[q]; a = A26[bad[q]]; b = A25[arg[q, k]] - pats[k]
    Ms.append(np.linalg.lstsq(np.c_[a, np.ones(50)], b, rcond=None)[0][:3].ravel())
Ms = np.array(Ms); cq = np.flatnonzero(conf)
# k-means with k=3 on M
rng = np.random.default_rng(0); C = Ms[rng.choice(len(Ms), 3, replace=False)]
for _ in range(50):
    lab = ((Ms[:, None] - C[None]) ** 2).sum(2).argmin(1)
    C = np.array([Ms[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])
for j in range(len(C)):
    print(f"M-cluster {j}: n {np.sum(lab == j)}\n{C[j].reshape(3, 3)}")
types = []
for j in range(len(C)):
    sel = cq[lab == j]
    if len(sel) < 5:
        continue
    ai = A26[bad[sel]]; bi = np.array([A25[arg[q, kb[q]]] for q in sel])
    types.append(fit(ai, bi))
print("initial types", len(types))
cand = {s: np.flatnonzero((s25 == s) & (l25 == L) & ~claimed) for s in np.unique(s26[bad])}
for it in range(6):
    best = np.full(len(bad), np.inf); bk = np.full(len(bad), -1); bj = np.full(len(bad), -1); sec = np.full(len(bad), np.inf)
    for s, c in cand.items():
        qq = np.flatnonzero(s26[bad] == s); Bf = A25[c].reshape(len(c), -1)
        for t, (M, N) in enumerate(types):
            P = (A26[bad[qq]] @ M + N).reshape(len(qq), -1)
            d = (P ** 2).sum(1)[:, None] + (Bf ** 2).sum(1)[None] - 2 * P @ Bf.T
            o = np.argsort(d, 1)[:, :2]; d1 = d[np.arange(len(qq)), o[:, 0]]; d2 = d[np.arange(len(qq)), o[:, 1]]
            upd = d1 < best[qq]
            sec[qq] = np.where(upd, np.minimum(best[qq], d2), np.minimum(sec[qq], d1))
            best[qq] = np.where(upd, d1, best[qq]); bk[qq] = np.where(upd, t, bk[qq]); bj[qq] = np.where(upd, c[o[:, 0]], bj[qq])
    rmsb = np.sqrt(np.maximum(best, 0) / 150); rmss = np.sqrt(np.maximum(sec, 0) / 150)
    print(f"it {it}: type counts {np.bincount(bk, minlength=len(types))} | best rms q {np.quantile(rmsb, [0.1, 0.5, 0.75, 0.9, 0.99]).round(5)} | second rms q {np.quantile(rmss, [0.1, 0.5]).round(4)}")
    new = []
    for t in range(len(types)):
        sel = np.flatnonzero((bk == t) & (rmsb < np.quantile(rmsb[bk == t], 0.8)))
        new.append(fit(A26[bad[sel]], A25[bj[sel]]) if len(sel) > 10 else types[t])
    types = new
for t, (M, N) in enumerate(types):
    U, S, Vt = np.linalg.svd(M)
    print(f"type {t}: n {np.sum(bk == t)}\nM=\n{M}\n sv {S.round(5)} det {np.linalg.det(M):.5f}\n N mean {N.mean(0).round(4)} N std {N.std(0).round(4)}")
np.savez(os.path.join(D, "p13_rglobal.npz"), bad=bad, bk=bk, bj=bj, best=best, sec=sec, Ms=np.array([t[0] for t in types]), Ns=np.array([t[1] for t in types]))
