"""Improved chain reconstruction: video LLR (+ inertial LLR for same-limb pairs) -> linear assignment -> cycle cutting.
Video-only chains are verified with the independent inertial boundary signal. Usage: python 06_chain_v2.py [partial]"""
import numpy as np, pandas as pd, os, sys, time
from scipy.optimize import linear_sum_assignment
from collections import Counter
D = r"E:\Claude code\wear\data\test"; A = r"E:\Claude code\wear\research\artifacts"
t0 = time.time()
meta = pd.read_csv(os.path.join(D, "test_meta_data.csv")); Xi = np.load(os.path.join(D, "test_inertial_data.npy"))
VP = os.path.join(D, "test_videomae_data.npy")
PARTIAL = len(sys.argv) > 1 and sys.argv[1] == "partial"
if PARTIAL:
    with open(VP, "rb") as f:
        ver = np.lib.format.read_magic(f); rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f); off = f.tell()
    K = min((os.path.getsize(VP) - off) // (int(np.prod(shape[1:])) * np.dtype(dtype).itemsize), shape[0])
    Vraw = np.memmap(VP, dtype=dtype, mode="r", offset=off, shape=(K,) + tuple(shape[1:])); meta = meta.iloc[:K].copy(); Xi = Xi[:K]
    print("PARTIAL rows", K)
else:
    Vraw = np.load(VP, mmap_mode="r")
V = np.ascontiguousarray(np.transpose(np.asarray(Vraw, dtype=np.float32), (0, 2, 1))); del Vraw
N = len(V); Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)
tail = Vn[:, 12:15].mean(1); head = Vn[:, 0:3].mean(1)
tail /= np.linalg.norm(tail, axis=1, keepdims=True); head /= np.linalg.norm(head, axis=1, keepdims=True)
# proxy for "true adjacent" video similarity: within-window first3 vs last3 (gap ~12 frames vs true cross-window gap 16)
h_in = Vn[:, 0:3].mean(1); h_in /= np.linalg.norm(h_in, axis=1, keepdims=True)
t_in = Vn[:, 12:15].mean(1); t_in /= np.linalg.norm(t_in, axis=1, keepdims=True)
within = (h_in * t_in).sum(1)
rng = np.random.default_rng(0)
limb_all = meta.sensor_location.astype(str).to_numpy(dtype=object)
step = np.linalg.norm(np.diff(Xi, axis=1), axis=2).ravel()          # intra-window sample steps (true-adjacent proxy for inertial gap)

def llr_from_samples(pos, neg, x, bins=60):
    lo = min(pos.min(), neg.min()); hi = max(np.percentile(pos, 99.9), np.percentile(neg, 99.9))
    e = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(pos, e); hn, _ = np.histogram(neg, e)
    hp = (hp + 1) / (hp.sum() + bins); hn = (hn + 1) / (hn.sum() + bins)
    k = np.clip(np.searchsorted(e, x, side="right") - 1, 0, bins - 1)
    return np.log(hp[k]) - np.log(hn[k])

summary = []
for s in sorted(meta.sbj_id.unique()):
    idx = meta.index[meta.sbj_id == s].values; n = len(idx); limbs = limb_all[idx]
    Sv = tail[idx] @ head[idx].T; np.fill_diagonal(Sv, -1)
    ra = rng.choice(n, 40000); rb = rng.choice(n, 40000); m = ra != rb
    neg_v = Sv[ra[m], rb[m]]; pos_v = within[idx]
    Lv = llr_from_samples(pos_v, neg_v, Sv.ravel()).reshape(n, n); np.fill_diagonal(Lv, -1e6)
    # inertial LLR for same-limb pairs
    last = Xi[idx, -1, :]; first = Xi[idx, 0, :]
    G = np.linalg.norm(last[:, None, :] - first[None, :, :], axis=2)         # n x n boundary gaps
    same = limbs[:, None] == limbs[None, :]
    neg_i = G[same & (ra[m][:, None] == ra[m][:, None])].ravel() if False else G[same].ravel()
    neg_i = rng.choice(neg_i, min(200000, len(neg_i)), replace=False)
    pos_i = np.linalg.norm(np.diff(Xi[idx], axis=1), axis=2).ravel()
    Li = np.zeros((n, n), np.float32)
    Li[same] = llr_from_samples(pos_i, neg_i, G[same])
    np.fill_diagonal(Li, 0)
    for name, L in (("video_only", Lv), ("video+inertial", Lv + Li)):
        # linear assignment: each window gets exactly one successor -> permutation (cycles). maximize L.
        r, c = linear_sum_assignment(-L)
        succ = c.copy()
        sc = L[r, c]
        # cut edges with low combined score, and cut the weakest edge of every cycle
        thr = np.percentile(sc, 3)
        succ[sc < thr] = -1
        # break cycles
        pred = np.full(n, -1); pred[succ[succ >= 0]] = np.where(succ >= 0)[0]
        seen = np.zeros(n, bool); ncyc = 0
        for st in range(n):
            if seen[st]: continue
            # walk back to a start (no pred) if exists, else it's a cycle
            cur = st; steps = 0
            while pred[cur] >= 0 and steps <= n:
                cur = pred[cur]; steps += 1
                if cur == st: break
            if pred[cur] >= 0:  # cycle
                # find weakest edge in cycle
                cyc = [cur]; x = succ[cur]
                while x != cur: cyc.append(x); x = succ[x]
                w = np.array([L[a, succ[a]] for a in cyc]); a = cyc[int(w.argmin())]
                pred[succ[a]] = -1; succ[a] = -1; ncyc += 1; cur = succ[a] if False else cyc[(int(w.argmin()) + 1) % len(cyc)]
            x = cur
            while x >= 0 and not seen[x]: seen[x] = True; x = succ[x]
        starts = np.where(pred < 0)[0]; chains = []
        for st in starts:
            ch = [st];
            while succ[ch[-1]] >= 0: ch.append(succ[ch[-1]])
            chains.append(ch)
        Lc = np.array([len(c) for c in chains])
        # inertial verification (independent only for video_only)
        eA = np.where(succ >= 0)[0]; eB = succ[eA]; sm = limbs[eA] == limbs[eB]
        g = G[eA[sm], eB[sm]]; smed = np.median(pos_i)
        frac_ok = (g < 2 * smed).mean(); frac_rand = (neg_i < 2 * smed).mean(); frac_true = (pos_i < 2 * smed).mean()
        prec_est = (frac_ok - frac_rand) / (frac_true - frac_rand)
        summary.append(dict(sbj=s, scoring=name, n=n, n_edges=len(eA), n_cycles_cut=ncyc, n_chains=len(chains), max_chain=Lc.max(),
                            median_chain=np.median(Lc), n_chains_ge_100=(Lc >= 100).sum(), frac_nodes_in_ge_100=Lc[Lc >= 100].sum() / n,
                            same_limb_edges=int(sm.sum()), edge_gap_median=np.median(g), frac_edge_gap_lt_2step=frac_ok,
                            frac_random_lt_2step=frac_rand, frac_trueadj_lt_2step=frac_true, est_edge_precision=prec_est))
        print(f"[{time.time()-t0:.0f}s] sbj {s} {name}: edges {len(eA)} cycles cut {ncyc} chains {len(chains)} max {Lc.max()} "
              f"| same-limb edges {sm.sum()}: gap med {np.median(g):.3f} frac<2step {frac_ok:.3f} (random {frac_rand:.3f}, true-adj proxy {frac_true:.3f}) -> est precision {prec_est:.2f}", flush=True)
        np.save(os.path.join(A, f"chainv2_succ_sbj{s}_{name.replace('+','_')}{'_partial' if PARTIAL else ''}.npy"), np.where(succ >= 0, idx[np.maximum(succ, 0)], -1))
        hist = sorted(Counter((np.minimum(Lc, 2000) // 50 * 50).tolist()).items())
        print("    chain-length histogram (bins of 50):", hist)
df = pd.DataFrame(summary); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(df.to_string()); df.to_csv(os.path.join(A, f"chainv2_summary{'_partial' if PARTIAL else ''}.csv"), index=False)
print("done %.0fs" % (time.time() - t0))
