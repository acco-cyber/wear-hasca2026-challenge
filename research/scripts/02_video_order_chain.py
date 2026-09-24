"""Tasks (2) and (3): is within-subject id order temporal? can temporal chains be reconstructed from
frame-level VideoMAE features? Plus independent inertial verification of video-chain edges.
Outputs artifacts under research/artifacts/.
"""
import numpy as np, pandas as pd, os, time, sys
from collections import Counter

D = r"E:\Claude code\wear\data\test"
A = r"E:\Claude code\wear\research\artifacts"
os.makedirs(A, exist_ok=True)
t0 = time.time()
meta = pd.read_csv(os.path.join(D, "test_meta_data.csv"))
Xi = np.load(os.path.join(D, "test_inertial_data.npy"))
VP = os.path.join(D, "test_videomae_data.npy")
PARTIAL = len(sys.argv) > 1 and sys.argv[1] == "partial"
if PARTIAL:
    # read header manually and memmap only the complete leading rows of a still-downloading file
    with open(VP, "rb") as f:
        ver = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        off = f.tell()
    rowbytes = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
    K = (os.path.getsize(VP) - off) // rowbytes
    K = min(K, shape[0])
    print(f"PARTIAL: header shape {shape} dtype {dtype} fortran {fortran} offset {off}; complete rows K={K}")
    Vraw = np.memmap(VP, dtype=dtype, mode="r", offset=off, shape=(K,) + tuple(shape[1:]), order="F" if fortran else "C")
    meta = meta.iloc[:K].copy(); Xi = Xi[:K]
else:
    Vraw = np.load(VP, mmap_mode="r")
print("raw video shape", Vraw.shape, Vraw.dtype, "fortran?", Vraw.flags['F_CONTIGUOUS'], "C?", Vraw.flags['C_CONTIGUOUS'])

# ---- axis semantics check on a sample: which axis is 'frames' (adjacent indices highly similar)?
smp = np.asarray(Vraw[:300], dtype=np.float32)  # (300, 768, 15)
def cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9); b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a * b).sum(-1)
# hypothesis H1: axis1=768 dims, axis2=15 frames -> frame vectors are smp[:, :, k]
h1 = cos(smp[:, :, :-1].transpose(0, 2, 1), smp[:, :, 1:].transpose(0, 2, 1)).mean()
# hypothesis H2: axis1=15?? no, sizes differ; instead check adjacent 'dims' similarity across the 15-vector
h2 = cos(smp[:, :-1, :], smp[:, 1:, :]).mean()
print(f"axis check: mean cos(adjacent frames under H1 [N,768,15]) = {h1:.4f}; mean cos(adjacent dims' 15-vectors) = {h2:.4f}")

# ---- load fully as (N,15,768) float32
V = np.ascontiguousarray(np.transpose(np.asarray(Vraw, dtype=np.float32), (0, 2, 1)))
del Vraw
N, T, Dm = V.shape
print("V shape", V.shape, "load time %.0fs" % (time.time() - t0))
print("global: mean %.4f std %.4f min %.3f max %.3f  NaN %s" % (V.mean(), V.std(), V.min(), V.max(), np.isnan(V).any()))
fn = np.linalg.norm(V, axis=2)
print("frame L2 norm: median %.3f p1 %.3f p99 %.3f; zero frames: %d" % (np.median(fn), np.percentile(fn, 1), np.percentile(fn, 99), (fn < 1e-6).sum()))

# exact duplicate video windows
h = {}; dups = []
for i in range(N):
    k = V[i].tobytes()
    if k in h: dups.append((h[k], i))
    else: h[k] = i
print("exact duplicate video windows:", len(dups))
# exact duplicate frames across windows (first frame hash)
hf = {}; fd = 0
for i in range(N):
    for t in (0, 14):
        k = V[i, t].tobytes()
        if k in hf and hf[k][0] != i: fd += 1
        else: hf[k] = (i, t)
print("exact duplicate boundary frames across windows:", fd)

# L2-normalised frames
Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)
# within-window adjacent-frame and far-frame similarity: calibration for what "true adjacency" looks like
adj = (Vn[:, :-1] * Vn[:, 1:]).sum(-1)          # gap 1 frame
far = (Vn[:, 0] * Vn[:, 14]).sum(-1)            # gap 14 frames (proxy for cross-window gap of 16 frames)
print("within-window cos: adjacent frames median %.4f p5 %.4f | frame0 vs frame14 median %.4f p5 %.4f p50 %.4f" % (
    np.median(adj), np.percentile(adj, 5), np.median(far), np.percentile(far, 5), np.percentile(far, 50)))

# mean-pooled
M = V.mean(1); Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
# concat-15
C = Vn.reshape(N, -1); Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
# tail/head descriptors
tail1 = Vn[:, 14]; head1 = Vn[:, 0]
tail3 = Vn[:, 12:15].mean(1); head3 = Vn[:, 0:3].mean(1)
tail3 /= np.linalg.norm(tail3, axis=1, keepdims=True); head3 /= np.linalg.norm(head3, axis=1, keepdims=True)

rng = np.random.default_rng(0)
rows2 = []; rows3 = []
chains_all = {}
edge_rows = []
for s in sorted(meta.sbj_id.unique()):
    idx = meta.index[meta.sbj_id == s].values
    n = len(idx)
    limbs = meta.sensor_location.values[idx]
    # ---------- (2) consecutive ids temporal?
    a = idx[:-1]; b = idx[1:]
    c_mean = (Mn[a] * Mn[b]).sum(1)
    c_lf = (tail1[a] * head1[b]).sum(1)
    ra = rng.choice(idx, 20000); rb = rng.choice(idx, 20000); m = ra != rb
    r_mean = (Mn[ra[m]] * Mn[rb[m]]).sum(1)
    r_lf = (tail1[ra[m]] * head1[rb[m]]).sum(1)
    rows2.append(dict(sbj=s, n=n, consec_meanpool_cos_median=np.median(c_mean), consec_meanpool_cos_mean=c_mean.mean(),
                      random_meanpool_cos_median=np.median(r_mean), random_meanpool_cos_mean=r_mean.mean(),
                      consec_last_first_cos_median=np.median(c_lf), random_last_first_cos_median=np.median(r_lf),
                      within_window_f0_f14_cos_median=np.median(far[idx])))
    # ---------- (3) chain reconstruction
    for name, TL, HD in (("last1_first1", tail1, head1), ("last3_first3", tail3, head3)):
        S = TL[idx] @ HD[idx].T                       # S[A,B] = sim(tail A, head B)
        np.fill_diagonal(S, -2)
        succ = S.argmax(1); best = S[np.arange(n), succ]
        S2 = S.copy(); S2[np.arange(n), succ] = -2; second = S2.max(1)
        pred = S.argmax(0)                            # best predecessor of each B
        mutual = (pred[succ] == np.arange(n))
        # greedy chaining: edges by descending sim, each node <=1 succ and <=1 pred, no cycles
        order = np.argsort(-S, axis=None)
        has_succ = np.full(n, -1); has_pred = np.full(n, -1)
        parent = np.arange(n)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        ne = 0; thr = -1.0
        for e in order[: n * 200]:
            A_, B_ = divmod(int(e), n)
            if A_ == B_ or has_succ[A_] >= 0 or has_pred[B_] >= 0: continue
            ra_, rb_ = find(A_), find(B_)
            if ra_ == rb_: continue
            has_succ[A_] = B_; has_pred[B_] = A_; parent[ra_] = rb_; ne += 1
            if ne == n - 1: break
        # chains
        starts = np.where(has_pred < 0)[0]
        chains = []
        for st in starts:
            ch = [st]; cur = st
            while has_succ[cur] >= 0:
                cur = has_succ[cur]; ch.append(cur)
            chains.append(ch)
        L = np.array([len(c) for c in chains])
        # independent inertial check on greedy edges with same limb
        eA = np.where(has_succ >= 0)[0]; eB = has_succ[eA]
        same = limbs[eA] == limbs[eB]
        gA = idx[eA[same]]; gB = idx[eB[same]]
        d_edge = np.linalg.norm(Xi[gA, -1, :] - Xi[gB, 0, :], axis=1)
        # random same-limb baseline
        rl = []
        for lb in np.unique(limbs):
            li = idx[limbs == lb]; x_ = rng.choice(li, 4000); y_ = rng.choice(li, 4000); mm = x_ != y_
            rl.append(np.linalg.norm(Xi[x_[mm], -1, :] - Xi[y_[mm], 0, :], axis=1))
        rl = np.concatenate(rl)
        intra = np.linalg.norm(np.diff(Xi[idx], axis=1), axis=2)
        intra_med = np.median(intra)
        # inertial check on MUTUAL edges only
        mA = np.where(mutual)[0]; mB = succ[mA]; msame = limbs[mA] == limbs[mB]
        d_mut = np.linalg.norm(Xi[idx[mA[msame]], -1, :] - Xi[idx[mB[msame]], 0, :], axis=1)
        rows3.append(dict(sbj=s, desc=name, n=n,
                          best_sim_median=np.median(best), best_sim_p10=np.percentile(best, 10), best_sim_p90=np.percentile(best, 90),
                          margin_median=np.median(best - second), margin_p10=np.percentile(best - second, 10),
                          frac_margin_gt_0_02=((best - second) > 0.02).mean(),
                          frac_mutual=mutual.mean(), n_chains=len(chains), max_chain=L.max(), median_chain=np.median(L),
                          n_chains_ge_100=(L >= 100).sum(), frac_nodes_in_chains_ge_100=L[L >= 100].sum() / n,
                          n_edges=ne, n_edges_same_limb=len(d_edge),
                          edge_inertial_gap_median=np.median(d_edge), random_inertial_gap_median=np.median(rl),
                          intra_step_median=intra_med,
                          frac_edge_gap_lt_2xstep=(d_edge < 2 * intra_med).mean(), frac_random_gap_lt_2xstep=(rl < 2 * intra_med).mean(),
                          frac_edge_gap_lt_0_1=(d_edge < 0.1).mean(), frac_random_gap_lt_0_1=(rl < 0.1).mean(),
                          n_mutual_same_limb=len(d_mut), mutual_edge_inertial_gap_median=np.median(d_mut) if len(d_mut) else np.nan,
                          frac_mutual_gap_lt_2xstep=(d_mut < 2 * intra_med).mean() if len(d_mut) else np.nan))
        chains_all[(s, name)] = chains
        if name == "last3_first3":
            np.save(os.path.join(A, f"chain_succ_sbj{s}_{name}.npy"), np.where(has_succ >= 0, idx[np.maximum(has_succ, 0)], -1))
            np.save(os.path.join(A, f"chain_bestsim_sbj{s}_{name}.npy"), best)
            for k, ch in enumerate(sorted(chains, key=len, reverse=True)[:5]):
                edge_rows.append(dict(sbj=s, chain_rank=k, length=len(ch), ids=" ".join(map(str, idx[ch][:30]))))
        print(f"[{time.time()-t0:.0f}s] sbj {s} {name}: best_sim med {np.median(best):.3f}, mutual {mutual.mean():.3f}, "
              f"chains {len(chains)} max {L.max()} | same-limb edges {len(d_edge)}: gap med {np.median(d_edge):.3f} vs random {np.median(rl):.3f} "
              f"(frac<2step {(d_edge < 2*intra_med).mean():.3f} vs {(rl < 2*intra_med).mean():.3f})", flush=True)
    # symmetric variants: mean-pooled and concat-15 : NN and mutual NN
    for name, F in (("sym_meanpool", Mn), ("sym_concat15", Cn)):
        S = F[idx] @ F[idx].T; np.fill_diagonal(S, -2)
        nn1 = S.argmax(1); b1 = S[np.arange(n), nn1]
        S2 = S.copy(); S2[np.arange(n), nn1] = -2; nn2 = S2.argmax(1); b2 = S2[np.arange(n), nn2]
        mutual = (nn1[nn1] == np.arange(n))
        # undirected greedy path building: degree<=2, no cycles
        order = np.argsort(-S, axis=None); deg = np.zeros(n, int); parent = np.arange(n)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        adjl = [[] for _ in range(n)]; ne = 0
        for e in order[: n * 400]:
            A_, B_ = divmod(int(e), n)
            if A_ >= B_ or deg[A_] >= 2 or deg[B_] >= 2: continue
            ra_, rb_ = find(A_), find(B_)
            if ra_ == rb_: continue
            parent[ra_] = rb_; deg[A_] += 1; deg[B_] += 1; adjl[A_].append(B_); adjl[B_].append(A_); ne += 1
            if ne == n - 1: break
        seen = np.zeros(n, bool); L = []
        for st in range(n):
            if seen[st] or deg[st] == 2 and True:
                if seen[st]: continue
            if deg[st] == 2: continue
            # walk path from endpoint
            cnt = 0; prev = -1; cur = st
            while True:
                seen[cur] = True; cnt += 1
                nxt = [x for x in adjl[cur] if x != prev]
                if not nxt: break
                prev, cur = cur, nxt[0]
            L.append(cnt)
        L = np.array(L) if L else np.array([0])
        # inertial check on undirected edges with same limb: either orientation could be right, take min of both gap orientations
        gaps = [];
        for A_ in range(n):
            for B_ in adjl[A_]:
                if A_ < B_ and limbs[A_] == limbs[B_]:
                    gAB = np.linalg.norm(Xi[idx[A_], -1] - Xi[idx[B_], 0]); gBA = np.linalg.norm(Xi[idx[B_], -1] - Xi[idx[A_], 0])
                    gaps.append(min(gAB, gBA))
        gaps = np.array(gaps)
        rows3.append(dict(sbj=s, desc=name, n=n, best_sim_median=np.median(b1), best_sim_p10=np.percentile(b1, 10), best_sim_p90=np.percentile(b1, 90),
                          margin_median=np.median(b1 - b2), margin_p10=np.percentile(b1 - b2, 10), frac_margin_gt_0_02=((b1 - b2) > 0.02).mean(),
                          frac_mutual=mutual.mean(), n_chains=len(L), max_chain=L.max(), median_chain=np.median(L),
                          n_chains_ge_100=(L >= 100).sum(), frac_nodes_in_chains_ge_100=L[L >= 100].sum() / n, n_edges=ne,
                          n_edges_same_limb=len(gaps), edge_inertial_gap_median=np.median(gaps) if len(gaps) else np.nan,
                          random_inertial_gap_median=np.median(rl), intra_step_median=intra_med,
                          frac_edge_gap_lt_2xstep=(gaps < 2 * intra_med).mean() if len(gaps) else np.nan,
                          frac_random_gap_lt_2xstep=(rl < 2 * intra_med).mean(),
                          frac_edge_gap_lt_0_1=(gaps < 0.1).mean() if len(gaps) else np.nan, frac_random_gap_lt_0_1=(rl < 0.1).mean(),
                          n_mutual_same_limb=np.nan, mutual_edge_inertial_gap_median=np.nan, frac_mutual_gap_lt_2xstep=np.nan))
        print(f"[{time.time()-t0:.0f}s] sbj {s} {name}: nn sim med {np.median(b1):.3f}, mutual {mutual.mean():.3f}, paths {len(L)} max {L.max()} | "
              f"same-limb edges {len(gaps)} gap med {np.median(gaps) if len(gaps) else -1:.3f} vs random {np.median(rl):.3f}", flush=True)

df2 = pd.DataFrame(rows2); df3 = pd.DataFrame(rows3)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("\n(2) consecutive-id similarity vs random:\n", df2.to_string())
print("\n(3) chain reconstruction stats:\n", df3.to_string())
df2.to_csv(os.path.join(A, "video_consec_similarity.csv"), index=False)
df3.to_csv(os.path.join(A, "video_chain_stats.csv"), index=False)
pd.DataFrame(edge_rows).to_csv(os.path.join(A, "video_chain_top_chains.csv"), index=False)
# chain length histograms for last3_first3
for s in sorted(meta.sbj_id.unique()):
    L = np.array([len(c) for c in chains_all[(s, "last3_first3")]])
    hist = Counter(np.minimum(L, 1000) // 10 * 10)
    print(f"sbj {s} last3_first3 chain-length histogram (bins of 10, capped 1000):", sorted(hist.items())[:25], "... longest:", sorted(L)[-8:])
np.save(os.path.join(A, "video_meanpool.npy"), M.astype(np.float32))
print("done %.0fs" % (time.time() - t0))
