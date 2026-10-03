"""Task 1 simulation: shuffle each limb's 1-s windows of a train session, re-group with the pair model + Hungarian.
usage: python sim.py <model_name> <session> [<session> ...]"""
import sys, os, time, json, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
import lightgbm as lgb
from feats import score_matrix
from common import PAIRS, PAIR_ID, LIMBS, hungarian

HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
CACHE = os.path.join(HERE, "cache")
model_name = sys.argv[1]
sessions = sys.argv[2:]


class Baseline:
    """Hand-weighted phase-free score on the pair-feature rows (no learning): intensity + spectral agreement."""
    def predict(self, X, **kw):
        from feats import ND, DESC_NAMES
        d = {n: 2 * ND + i for i, n in enumerate(DESC_NAMES)}
        return (-2.0 * X[:, d["logstd"]] - 1.0 * X[:, d["logjerk"]] - 0.15 * X[:, d["centroid"]]
                - 0.3 * X[:, d["fdom"]] + 2.0 * X[:, 3 * ND] + 1.0 * X[:, 3 * ND + 2]).astype(np.float32)


bst = Baseline() if model_name == "baseline" else lgb.Booster(model_file=os.path.join(HERE, model_name + ".txt"))
# optional within-limb neighbourhood smoothing of the score matrices: S' = S + lam * P_a S P_b^T
_sm = os.environ.get("SMOOTH", "0:0").split(":")
SMOOTH_K, SMOOTH_LAM = int(_sm[0]), float(_sm[1])
tag = model_name + (f"_s{SMOOTH_K}x{SMOOTH_LAM:g}" if SMOOTH_K > 0 else "")
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, identity


def knn_matrix(desc, k):
    X = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)
    n = len(X)
    Dm = cdist(X, X); np.fill_diagonal(Dm, np.inf)
    nn = np.argpartition(Dm, k, axis=1)[:, :k]
    P = csr_matrix((np.full(n * k, 1.0 / (k + 1), np.float32), (np.repeat(np.arange(n), k), nn.ravel())), shape=(n, n))
    return (P + identity(n, dtype=np.float32, format="csr") * (1.0 / (k + 1))).tocsr()


def smooth_scores(S, Pa, Pb, lam):
    return (S + lam * np.asarray(Pa @ np.asarray((Pb @ S.T)).T)).astype(np.float32)
rng = np.random.default_rng(123)


def load_cache(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    D = [{k: z[f"{k}{l}"] for k in ("desc", "z_spec", "z_specj", "z_acf", "std", "dyn", "gmean")} for l in range(4)]
    return D, z["lab"]


def sub(D, idx):
    return {k: v[idx] for k, v in D.items()}


def acc_split(ok, same, mask):
    return dict(exact=float(ok[mask].mean()) if mask.any() else float("nan"),
                same=float(same[mask].mean()) if mask.any() else float("nan"), n=int(mask.sum()))


def pair_score(S, a, b, ia, ib):
    """score between limb-a rows ia and limb-b rows ib using canonical matrices S[(lo,hi)] (lo<hi)."""
    if a < b:
        return S[(a, b)][ia, ib]
    return S[(b, a)][ib, ia]


results = {}
calib_scores, calib_same = [], []  # (assigned-pair log-odds, same-label flag) from the anchor-centric assignment
for name in sessions:
    t0 = time.time()
    D, lab = load_cache(name)
    N = len(lab)
    # shuffle each limb independently; perm[l][r] = true second of shuffled row r
    perm = [rng.permutation(N) for _ in range(4)]
    inv = [np.argsort(p) for p in perm]  # inv[l][s] = shuffled row of second s
    Ds = [sub(D[l], perm[l]) for l in range(4)]
    labs = [lab[perm[l]] for l in range(4)]
    dyns = [D[l]["dyn"][perm[l]] for l in range(4)]
    # chance same-label rate
    _, cnt = np.unique(lab, return_counts=True)
    chance = float(((cnt / N) ** 2).sum())
    res = {"N": int(N), "chance_same": chance, "pairs": {}}
    S = {}
    for (a, b) in PAIRS:
        S[(a, b)] = score_matrix(bst, Ds[a], Ds[b], PAIR_ID[(a, b)])
    print(f"{name}: N={N} scores in {time.time()-t0:.0f}s", flush=True)
    if SMOOTH_K > 0:
        P = [knn_matrix(Ds[l]["desc"], SMOOTH_K) for l in range(4)]
        for (a, b) in PAIRS:
            S[(a, b)] = smooth_scores(S[(a, b)], P[a], P[b], SMOOTH_LAM)
        print(f"  smoothed scores with k={SMOOTH_K} lam={SMOOTH_LAM}", flush=True)
    for (a, b) in PAIRS:
        asg = hungarian(S[(a, b)])  # for shuffled a-row r -> shuffled b-row
        ok = perm[b][asg] == perm[a]
        same = labs[b][asg] == labs[a]
        top1 = perm[b][S[(a, b)].argmax(1)] == perm[a]
        r = {k: acc_split(ok, same, m) for k, m in [("all", np.ones(N, bool)), ("dyn", dyns[a]), ("sta", ~dyns[a])]}
        r["top1_exact"] = float(top1.mean())
        res["pairs"][f"{LIMBS[a]}-{LIMBS[b]}"] = r
        print(f"  {LIMBS[a]:9s}-{LIMBS[b]:9s} exact all {r['all']['exact']:.3f} dyn {r['dyn']['exact']:.3f} sta {r['sta']['exact']:.3f} | "
              f"same-label all {r['all']['same']:.3f} dyn {r['dyn']['same']:.3f} sta {r['sta']['same']:.3f} (chance {chance:.3f}) | n_dyn {r['dyn']['n']} n_sta {r['sta']['n']} top1 {r['top1_exact']:.3f}", flush=True)
    # ---- consistent 4-way grouping: legs L<->R, arms L<->R, then leg-pair <-> arm-pair
    leg = hungarian(S[(1, 3)])   # LL row i -> RL row leg[i]
    arm = hungarian(S[(0, 2)])   # LA row i -> RA row arm[i]
    # S4[i (leg pair i, keyed by LL row i), j (arm pair j, keyed by LA row j)]
    S4 = S[(0, 1)].T + S[(0, 3)][:, leg].T + S[(1, 2)][:, arm] + S[(2, 3)][arm][:, leg].T
    g = hungarian(S4)  # LL row i -> LA row g[i]
    grp = np.stack([g, np.arange(N), arm[g], leg], 1)  # rows per limb: LA, LL, RA, RL
    secs = np.stack([perm[l][grp[:, l]] for l in range(4)], 1)
    glab = np.stack([labs[l][grp[:, l]] for l in range(4)], 1)
    all_exact = (secs == secs[:, :1]).all(1)
    all_same = (glab == glab[:, :1]).all(1)
    dyn_any = np.stack([dyns[l][grp[:, l]] for l in range(4)], 1).any(1)
    res["group4"] = {"exact4": float(all_exact.mean()), "same4": float(all_same.mean()),
                     "same4_dyn": float(all_same[dyn_any].mean()), "same4_sta": float(all_same[~dyn_any].mean()),
                     "pairwise_same": float((glab[:, :, None] == glab[:, None, :]).mean())}
    print(f"  4-way: exact {res['group4']['exact4']:.3f} all-same-label {res['group4']['same4']:.3f} (dyn {res['group4']['same4_dyn']:.3f} sta {res['group4']['same4_sta']:.3f}) pairwise-same {res['group4']['pairwise_same']:.3f}", flush=True)
    # ---- anchor simulation: one known limb per second, assign the other three (anchor-centric Hungarian + refinement)
    anc = rng.integers(0, 4, size=N)  # anchor limb per true second
    anc_row = np.array([inv[anc[s]][s] for s in range(N)])  # shuffled row of the anchor window
    groups = np.full((N, 4), -1, dtype=np.int64)
    groups[np.arange(N), anc] = anc_row
    for it in range(3):
        new = groups.copy()
        for m in range(4):
            secs_m = np.where(anc != m)[0]              # seconds that need a limb-m window
            rows_m = np.where(anc[perm[m]] != m)[0]      # shuffled limb-m rows that are not anchors (true set unknown to method: emulate via anchor set complement)
            # NOTE: the method knows which limb-m rows are anchors (they are the identified tiles); the complement is what it must place.
            C = np.zeros((len(rows_m), len(secs_m)), dtype=np.float32)
            for L in range(4):
                if L == m: continue
                sel = np.where(anc[secs_m] == L)[0]
                if len(sel) == 0: continue
                arow = anc_row[secs_m[sel]]
                if m < L:
                    C[:, sel] = S[(m, L)][rows_m][:, arow]
                else:
                    C[:, sel] = S[(L, m)][arow][:, rows_m].T
                if it > 0:  # add consistency with the other already-assigned non-anchor limbs
                    for K in range(4):
                        if K in (m, L): continue
                        krow = groups[secs_m[sel], K]
                        if m < K:
                            C[:, sel] += 0.5 * S[(m, K)][rows_m][:, krow]
                        else:
                            C[:, sel] += 0.5 * S[(K, m)][krow][:, rows_m].T
            asg = hungarian(C)  # row k -> column asg[k]
            new[secs_m[asg], m] = rows_m
        groups = new
        # evaluate: for each second, the 3 non-anchor limbs
        ex = []; sm = []; dy = []
        for m in range(4):
            sel = np.where(anc != m)[0]
            ex.append(perm[m][groups[sel, m]] == sel)
            sm.append(labs[m][groups[sel, m]] == lab[sel])
            dy.append(np.array([D[anc[s]]["dyn"][s] for s in sel]))
        ex = np.concatenate(ex); sm = np.concatenate(sm); dy = np.concatenate(dy)
        r = {k: acc_split(ex, sm, mk) for k, mk in [("all", np.ones(len(ex), bool)), ("dyn", dy), ("sta", ~dy)]}
        res[f"anchor_iter{it}"] = r
        print(f"  anchor-centric iter{it}: exact {r['all']['exact']:.3f} same-label all {r['all']['same']:.3f} dyn {r['dyn']['same']:.3f} sta {r['sta']['same']:.3f}", flush=True)
    # calibration data: log-odds of (assigned row, anchor row) vs same-label outcome
    for m in range(4):
        sel = np.where(anc != m)[0]
        sc = np.array([pair_score(S, m, anc[s], groups[s, m], anc_row[s]) for s in sel], dtype=np.float32)
        calib_scores.append(sc); calib_same.append(labs[m][groups[sel, m]] == lab[sel])
    # per-activity same-label rate of the final anchor-centric grouping (diagnostic)
    per_act = {}
    for m in range(4):
        sel = np.where(anc != m)[0]
        for s, ok in zip(sel, labs[m][groups[sel, m]] == lab[sel]):
            per_act.setdefault(lab[s], []).append(ok)
    res["anchor_per_activity"] = {k: [float(np.mean(v)), len(v)] for k, v in per_act.items()}
    print("  per-activity same-label (anchor-centric): " + ", ".join(f"{k[:14]}={np.mean(v):.2f}" for k, v in sorted(per_act.items())), flush=True)
    results[name] = res
    print(f"  done {name} in {time.time()-t0:.0f}s", flush=True)

with open(os.path.join(HERE, f"sim_{tag}.json"), "w") as f:
    json.dump(results, f, indent=1)
# binned calibration table: same-label rate as a function of pair log-odds
sc = np.concatenate(calib_scores); sm = np.concatenate(calib_same).astype(np.float32)
edges = np.quantile(sc, np.linspace(0, 1, 21)); edges[0] = -np.inf; edges[-1] = np.inf
b = np.clip(np.searchsorted(edges, sc, side="right") - 1, 0, 19)
rate = np.array([sm[b == k].mean() if (b == k).any() else np.nan for k in range(20)])
np.savez(os.path.join(HERE, f"calib_{tag}.npz"), edges=edges, rate=rate)
print("calibration (log-odds bin -> same-label rate):", np.round(rate, 2))
print("saved")
