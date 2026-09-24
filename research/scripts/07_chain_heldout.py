"""Held-out validation of video+inertial chaining: inertial LLR uses only the x,y boundary gap; verification uses the
z-axis boundary gap (|last_z[A]-first_z[B]|) against intra-window z steps (true-adjacent proxy) and random same-limb pairs."""
import numpy as np, pandas as pd, os, time
from scipy.optimize import linear_sum_assignment
from collections import Counter
D = r"E:\Claude code\wear\data\test"; A = r"E:\Claude code\wear\research\artifacts"
t0 = time.time(); rng = np.random.default_rng(1)
meta = pd.read_csv(os.path.join(D, "test_meta_data.csv")); Xi = np.load(os.path.join(D, "test_inertial_data.npy"))
V = np.ascontiguousarray(np.transpose(np.asarray(np.load(os.path.join(D, "test_videomae_data.npy"), mmap_mode="r"), dtype=np.float32), (0, 2, 1)))
Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)
tail = Vn[:, 12:15].mean(1); head = Vn[:, 0:3].mean(1); tail /= np.linalg.norm(tail, axis=1, keepdims=True); head /= np.linalg.norm(head, axis=1, keepdims=True)
within = (head * tail).sum(1)
limb_all = meta.sensor_location.astype(str).to_numpy(dtype=object)
def llr(pos, neg, x, bins=60):
    lo = min(pos.min(), neg.min()); hi = max(np.percentile(pos, 99.9), np.percentile(neg, 99.9)); e = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(pos, e); hn, _ = np.histogram(neg, e); hp = (hp + 1) / (hp.sum() + bins); hn = (hn + 1) / (hn.sum() + bins)
    k = np.clip(np.searchsorted(e, x, side="right") - 1, 0, bins - 1); return np.log(hp[k]) - np.log(hn[k])
def lsa_chain(L, cut_pct=3):
    n = len(L); r, c = linear_sum_assignment(-L); succ = c.copy(); sc = L[r, c]; succ[sc < np.percentile(sc, cut_pct)] = -1
    pred = np.full(n, -1); pred[succ[succ >= 0]] = np.where(succ >= 0)[0]; seen = np.zeros(n, bool)
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
    return succ, pred
rows = []; allsucc = np.full(len(meta), -1)
for s in sorted(meta.sbj_id.unique()):
    idx = meta.index[meta.sbj_id == s].values; n = len(idx); limbs = limb_all[idx]
    Sv = tail[idx] @ head[idx].T; np.fill_diagonal(Sv, -1)
    ra = rng.choice(n, 40000); rb = rng.choice(n, 40000); m = ra != rb
    Lv = llr(within[idx], Sv[ra[m], rb[m]], Sv.ravel()).reshape(n, n); np.fill_diagonal(Lv, -1e6)
    last = Xi[idx, -1, :]; first = Xi[idx, 0, :]; same = limbs[:, None] == limbs[None, :]
    Gxy = np.linalg.norm(last[:, None, :2] - first[None, :, :2], axis=2); Gz = np.abs(last[:, None, 2] - first[None, :, 2])
    pos_xy = np.linalg.norm(np.diff(Xi[idx][:, :, :2], axis=1), axis=2).ravel(); pos_z = np.abs(np.diff(Xi[idx][:, :, 2], axis=1)).ravel()
    neg_xy = rng.choice(Gxy[same].ravel(), 200000, replace=False); neg_z = rng.choice(Gz[same].ravel(), 200000, replace=False)
    Li = np.zeros((n, n), np.float32); Li[same] = llr(pos_xy, neg_xy, Gxy[same]); np.fill_diagonal(Li, 0)
    zthr = 2 * np.median(pos_z); p_true = (pos_z < zthr).mean(); p_rand = (neg_z < zthr).mean()
    for name, L in (("video_only", Lv), ("video+xy_inertial", Lv + Li)):
        succ, pred = lsa_chain(L); e = np.where(succ >= 0)[0]; sm = limbs[e] == limbs[succ[e]]
        gz = Gz[e[sm], succ[e[sm]]]; p_edge = (gz < zthr).mean(); prec = (p_edge - p_rand) / (p_true - p_rand)
        starts = np.where(pred < 0)[0]; Lc = []
        for st in starts:
            k = 1; x = st
            while succ[x] >= 0: x = succ[x]; k += 1
            Lc.append(k)
        Lc = np.array(Lc)
        # also: precision by video-score quantile (are high-score edges more reliable?)
        sc = L[e, succ[e]]; q = np.percentile(sc, [50, 80]); hi = sc[sm] >= q[1]; lo = sc[sm] < q[0]
        prec_hi = ((gz[hi] < zthr).mean() - p_rand) / (p_true - p_rand) if hi.sum() > 20 else np.nan
        prec_lo = ((gz[lo] < zthr).mean() - p_rand) / (p_true - p_rand) if lo.sum() > 20 else np.nan
        rows.append(dict(sbj=s, scoring=name, n=n, edges=len(e), chains=len(Lc), longest=Lc.max(), nodes_in_chains_ge_100=Lc[Lc >= 100].sum() / n,
                         same_limb_edges=int(sm.sum()), z_gap_median=np.median(gz), frac_z_lt_thr=p_edge, random_z_lt_thr=p_rand, trueadj_z_lt_thr=p_true,
                         est_precision_heldout_z=prec, est_prec_top20pct_score=prec_hi, est_prec_bottom50pct_score=prec_lo))
        print(f"[{time.time()-t0:.0f}s] sbj {s} {name}: edges {len(e)} chains {len(Lc)} longest {Lc.max()} | held-out z-gap check on {sm.sum()} same-limb edges: "
              f"frac<thr {p_edge:.3f} (random {p_rand:.3f}, true-adj {p_true:.3f}) -> precision {prec:.2f} | top-20% score edges {prec_hi:.2f}, bottom-50% {prec_lo:.2f}", flush=True)
        if name != "video_only": allsucc[idx[e]] = idx[succ[e]]
df = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); print(df.to_string())
df.to_csv(os.path.join(A, "chain_heldout_summary.csv"), index=False)
pd.DataFrame(dict(id=meta.id, sbj_id=meta.sbj_id, succ_id=allsucc)).to_csv(os.path.join(A, "chain_successor_video_xyinertial.csv"), index=False)
print("done %.0fs" % (time.time() - t0))
