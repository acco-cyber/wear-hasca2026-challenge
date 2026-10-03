"""Task 2: group the shuffled w25 limb windows per test subject into seconds, anchored on the identified tiles.
usage: python run_test.py <model_name>      env: SMOOTH=k:lam REFINE=w CONF_THR=0.8
Uses cached score matrices scores/<model>_test<sbj>.npz (computed by score_cache.py) and calib_<tag>.npz (from eval_cfg.py).
Writes test_groups.npz: tile (12234,), w25_rows (12234,4) [limb order LA,LL,RA,RL; -1 = unassigned], conf (12234,4),
conf_raw (12234,4) (confidence before thresholding), anchor_limb (12234,)."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from feats import describe, score_matrix
from common import PAIRS, PAIR_ID, LIMBS
from grouping import knn_matrix, smooth_scores, anchor_assign, finish_feats, is_sym_pair, apply_calib

HERE = r"E:\Claude code\wear\exp\hyb\agents\group"; SC = os.path.join(HERE, "scores")
model_name = sys.argv[1] if len(sys.argv) > 1 else "pair_sec"
_sm = os.environ.get("SMOOTH", "0:0").split(":"); SMOOTH_K, SMOOTH_LAM = int(_sm[0]), float(_sm[1])
REFINE = float(os.environ.get("REFINE", "0.5")); CONF_THR = float(os.environ.get("CONF_THR", "0.8"))
tag = model_name + (f"_s{SMOOTH_K}x{SMOOTH_LAM:g}" if SMOOTH_K > 0 else "") + (f"_r{REFINE:g}" if REFINE != 0.5 else "")
cal = np.load(os.path.join(HERE, f"calib_{tag}.npz"))
print("config", tag, "conf threshold", CONF_THR, flush=True)
LIMB_IDX = {n: i for i, n in enumerate(LIMBS)}

w = np.load(r"E:\Claude code\wear\work\w25\w25.npz")
acc, wsbj, wlimb = w["acc"], w["sbj"], w["limb"]
m = np.load(r"E:\Claude code\wear\work\w25\match.npz")
twin, corr = m["twin"], m["corr"]
meta = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
tiles = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy").astype(np.float32)
T = len(meta)
tile_limb = meta.sensor_location.map(LIMB_IDX).to_numpy(); tile_sbj = meta.sbj_id.to_numpy()
assert (wsbj[twin] == tile_sbj).all() and (wlimb[twin] == tile_limb).all(), "twin meta mismatch"
maxdiff = np.abs(np.nan_to_num(tiles) - np.nan_to_num(acc[twin])).max(axis=(1, 2))
exact = maxdiff < 1e-4
uniq, inv, cnt = np.unique(twin, return_inverse=True, return_counts=True)
dup = cnt[inv] > 1
# a tile is a reliable anchor if its twin is an exact copy and no other tile claims the same w25 row
# (if several tiles share a row and exactly one is an exact copy, that one keeps it)
reliable = exact & ~dup
for u in uniq[cnt > 1]:
    idx = np.where(twin == u)[0]; ex_idx = idx[exact[idx]]
    if len(ex_idx) == 1:
        reliable[ex_idx[0]] = True
print(f"tiles {T}; twins that are exact copies (maxabs<1e-4): {exact.sum()}; corr>0.99: {(corr > 0.99).sum()}; "
      f"tiles sharing a twin row with another tile: {dup.sum()} ({len(uniq[cnt > 1])} rows); reliable anchors: {reliable.sum()}; "
      f"maxdiff quantiles(.5,.9,.99,1) {np.round(np.quantile(maxdiff, [0.5, 0.9, 0.99, 1.0]), 4)}", flush=True)

out_rows = np.full((T, 4), -1, dtype=np.int64)
conf_raw = np.zeros((T, 4), dtype=np.float32)
for sb in np.unique(tile_sbj):
    t0 = time.time()
    tsel = np.where(tile_sbj == sb)[0]; N = len(tsel)
    rows = [np.where((wsbj == sb) & (wlimb == l))[0] for l in range(4)]
    assert all(len(r) == N for r in rows), (sb, N, [len(r) for r in rows])
    D = [describe(acc[r]) for r in rows]
    pos = [dict(zip(r.tolist(), range(len(r)))) for r in rows]
    anc = tile_limb[tsel]
    anc_loc = np.array([pos[anc[k]][twin[tsel[k]]] for k in range(N)])
    valid = reliable[tsel]
    f = os.path.join(SC, f"{model_name}_test{sb}.npz")
    if os.path.exists(f):
        sz = np.load(f); S = {(a, b): sz[f"S{a}{b}"].astype(np.float32) for (a, b) in PAIRS}
    else:
        import lightgbm as lgb
        bst = lgb.Booster(model_file=os.path.join(HERE, model_name + ".txt"))
        S = {(a, b): score_matrix(bst, D[a], D[b], PAIR_ID[(a, b)]) for (a, b) in PAIRS}
    if SMOOTH_K > 0:
        P = [knn_matrix(D[l]["desc"], SMOOTH_K) for l in range(4)]
        S = {p: smooth_scores(S[p], P[p[0]], P[p[1]], SMOOTH_LAM) for p in PAIRS}
    is_anchor = [np.zeros(N, bool) for _ in range(4)]
    for k in np.where(valid)[0]:
        is_anchor[anc[k]][anc_loc[k]] = True
    groups, feats = anchor_assign(S, anc, anc_loc, is_anchor, refine=REFINE, valid=valid)
    feats = finish_feats(S, groups, feats, anc, [D[l]["dyn"] for l in range(4)], valid=valid)
    for k in range(N):
        for l in range(4):
            if groups[k, l] >= 0:
                out_rows[tsel[k], l] = rows[l][groups[k, l]]
    # the tile's own slot: its twin (conf 1 if reliable exact copy, else the match corr, flagged by not being 1.0)
    out_rows[tsel, anc] = twin[tsel]
    conf_raw[tsel, anc] = np.where(valid, 1.0, np.minimum(corr[tsel], 0.999))
    for mm in range(4):
        sel = np.where(valid & (anc != mm))[0]
        sym = np.array([is_sym_pair(mm, a) for a in anc[sel]])
        conf_raw[tsel[sel], mm] = apply_calib(cal, feats[sel, mm, :], sym)
    print(f"sbj{sb}: N={N} anchors per limb {np.bincount(anc, minlength=4).tolist()} (reliable {valid.sum()}); "
          f"mean conf per non-anchor slot {np.round([conf_raw[tsel[valid & (anc != l)], l].mean() for l in range(4)], 3)} ({time.time()-t0:.0f}s)", flush=True)

# threshold: low-confidence limbs are left unassigned (-1)
conf = conf_raw.copy()
keep = (conf_raw >= CONF_THR) & (out_rows >= 0)
keep[np.arange(T), tile_limb] = True  # the tile's own twin is always kept (conf = 1 if reliable, else the match corr)
w25_rows = np.where(keep, out_rows, -1)
conf[~keep] = 0.0
anchor_limb = tile_limb
n_lim = keep.sum(1)
frac = np.bincount(n_lim, minlength=5)[1:5] / T
assigned_rows = w25_rows[w25_rows >= 0]
print(f"\nthreshold {CONF_THR}: tiles with 1/2/3/4 limbs assigned: {np.round(frac, 3).tolist()}  (counts {np.bincount(n_lim, minlength=5)[1:5].tolist()})")
print(f"w25 rows assigned: {len(assigned_rows)} (unique {len(np.unique(assigned_rows))}); left unassigned: {len(acc) - len(np.unique(assigned_rows))} of {len(acc)}")
for l in range(4):
    non_anchor = anchor_limb != l
    print(f"  slot {LIMBS[l]:9s}: non-anchor tiles {non_anchor.sum()}, assigned {keep[non_anchor, l].sum()} ({keep[non_anchor, l].mean():.3f}), "
          f"mean conf of assigned {conf[non_anchor & keep[:, l], l].mean() if (non_anchor & keep[:, l]).any() else float('nan'):.3f}")
for sb in np.unique(tile_sbj):
    s = tile_sbj == sb
    print(f"  sbj{sb}: tiles with 1/2/3/4 limbs {np.round(np.bincount(n_lim[s], minlength=5)[1:5] / s.sum(), 3).tolist()}")
np.savez(os.path.join(HERE, "test_groups.npz"), tile=np.arange(T), w25_rows=w25_rows, conf=conf, conf_raw=conf_raw,
         w25_rows_unthresholded=out_rows, anchor_limb=anchor_limb, limb_order=np.array(LIMBS), model=tag, conf_thr=CONF_THR)
print("saved", os.path.join(HERE, "test_groups.npz"))
