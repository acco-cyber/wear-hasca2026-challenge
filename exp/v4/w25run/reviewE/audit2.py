"""Review E, part 2: train/test feature mismatch for UNANCHORED tiles in the 'deaug' simulation.
In build_test_links.sim_table every tile keeps its TRUE chain position pos[i] even when it is unanchored, while the real
test (world_table) sets pos = 0 for unanchored tiles. tdlib.subject_table masks dirsame only by anch_A, so for a same-limb
pair A (anchored) -> B (unanchored) the simulation feature dirsame = [chain succ of A == true node of B] carries information
the test can never have (there it is [chain succ of A == node 0] ~ 0).
This script re-scores the NESTED fold models of a finished testD run on held-out training subjects twice:
  'as_trained' : pos as in sim_table (must reproduce the run's OOF links exactly = harness check)
  'test_like'  : pos[~anch] = 0 exactly as world_table does on the real test
and reports exact-successor rates (overall / into unanchored B / same-limb), i.e. how much the OOF links overstate the
test-side link quality and how test-side decisions for unanchored tiles differ.
  python audit2.py --fit K7 [--mode deaug]"""
import os, sys, time, argparse, json
import numpy as np
TD = r"E:\Claude code\wear\exp\v4\w25run\testD"
sys.path.insert(0, TD)
from tdlib import *                                                   # noqa: E402  (read-only use of testD helpers)
import lightgbm as lgb                                                # noqa: E402
from joblib import Parallel, delayed                                  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--fit", default="K7"); ap.add_argument("--mode", default="deaug"); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--subjects", default="")
a = ap.parse_args()
T0 = time.time()
HERE_E = os.path.dirname(os.path.abspath(__file__))
TAG = f"{a.fit}_{a.mode}"; FIT = FITS[a.fit]
S = load_stage(FIT); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]; fold = S["oof_fold"]; N = len(sbj)
fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
LZP = os.path.join(FIT, "link_logodds.npz")
subs = [int(x) for x in a.subjects.split(",")] if a.subjects else [int(x) for x in np.unique(sbj)]
ref_oof = np.load(os.path.join(TD, f"links_oof_{TAG}.npz"))["oof_succ"][0]


def table(s, zero_unanch):
    loc = np.flatnonzero(sbj == s); n = len(loc)
    rows = loc[np.lexsort((S["oof_start"][loc], S["oof_rec"][loc]))]
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    pos = p_of[loc]; lim = sens[loc]
    z = np.load(os.path.join(TD, "cache", f"chain{a.mode}_s{s}.npz")); assert (z["rows"] == rows).all()
    anch = ~z["aug"][lim, pos]
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    if zero_unanch:
        pos = pos.copy(); pos[~anch] = 0
    tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
    chains = [chain_tuple(z[f"succ{L}"], z[f"conf{L}"]) for L in range(4)]
    LZ = np.load(LZP); cand, Lo = topk_cands(LZ[f"oof_{s}_cand"], LZ[f"oof_{s}_L"], 50)
    t = subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, tl, 16); t["loc"] = loc
    return t


def run_subject(s):
    f = fold_of[s]
    m1 = lgb.Booster(model_file=os.path.join(TD, "models", f"{TAG}_s1_f{f}.txt")); m2 = lgb.Booster(model_file=os.path.join(TD, "models", f"{TAG}_s2_f{f}.txt"))
    out = {}
    for variant, zu in (("as_trained", False), ("test_like", True)):
        t = table(s, zu); names = t["names"]; cols1 = feat_cols(names); cols2 = feat_cols(names + GAP_NAMES)
        base = t["X"][:, names.index("lo")].astype(np.float64)
        p1 = (m1.predict(t["X"][:, cols1], raw_score=True, num_threads=1) + base).astype(np.float32)
        _, r1 = match(t, p1, 1, s)
        g = gap_feats(t, r1[0][0], 16)
        Xk = np.concatenate([t["X"], g, p1[:, None]], 1)
        p2 = (m2.predict(Xk[:, cols2], raw_score=True, num_threads=1) + p1).astype(np.float64).astype(np.float32)
        _, r2 = match(t, p2, 1, s)
        su = r2[0][0]; loc = t["loc"]
        out[variant] = np.where(su >= 0, loc[np.maximum(su, 0)], -1)
        if variant == "as_trained":
            ds = t["X"][:, names.index("dirsame")]; same = t["X"][:, names.index("same_limb")] == 1
            aA = t["anch"][t["pi"]]; aB = t["anch"][t["pj"]]; y = t["y"] == 1
            k = same & aA & ~aB
            out["dirsame_stats"] = (int(np.sum(k & y)), int(np.sum(k & y & (ds == 1))), int(np.sum(k & ~y)), int(np.sum(k & ~y & (ds == 1))))
    out["anch"] = (t["loc"], t["anch"])
    return s, out


res = Parallel(n_jobs=a.jobs)(delayed(run_subject)(s) for s in sorted(subs, key=lambda s: -(sbj == s).sum()))
SU = {v: np.full(N, -1, np.int64) for v in ("as_trained", "test_like")}; AN = np.ones(N, bool); dstat = np.zeros(4, int)
for s, o in res:
    loc, an = o["anch"]; AN[loc] = an; dstat += np.array(o["dirsame_stats"])
    for v in SU:
        SU[v][loc] = o[v]
mask = np.isin(sbj, subs); h = (ts >= 0) & mask
B_un = np.zeros(N, bool); B_un[h] = ~AN[np.maximum(ts, 0)][h]
same = np.zeros(N, bool); same[h] = sens[np.maximum(ts, 0)][h] == sens[h]
r = {}
print(f"[{time.time() - T0:.0f}s] {TAG} subjects {len(subs)}: harness check as_trained == run's OOF links: {np.mean(SU['as_trained'][mask] == ref_oof[mask]):.4f}")
print(f"   sim pairs same-limb, A anchored, B unanchored: positives {dstat[0]} with dirsame=1 {dstat[1]} ({dstat[1] / max(dstat[0], 1):.3f}); negatives {dstat[2]} with dirsame=1 {dstat[3]} ({dstat[3] / max(dstat[2], 1):.4f})")
for v in SU:
    e = SU[v][h] == ts[h]
    r[v] = dict(exact=float(e.mean()), exact_into_unanch_B=float(e[B_un[h]].mean()), exact_into_unanch_B_same=float(e[(B_un & same)[h]].mean()),
                exact_into_anch_B=float(e[~B_un[h]].mean()), n_unanch_same=int((B_un & same)[h].sum()))
    # how many A link INTO an unanchored same-limb B (label free, comparable with the test)
    su = SU[v]; lk = mask & (su >= 0)
    r[v]["share_links_into_unanch_same"] = float(np.mean((~AN[np.maximum(su, 0)] & (sens[np.maximum(su, 0)] == sens))[lk]))
    print(f"   {v:11s}: OOF exact {r[v]['exact']:.4f} | into unanchored B {r[v]['exact_into_unanch_B']:.4f} (same-limb {r[v]['exact_into_unanch_B_same']:.4f}, n {r[v]['n_unanch_same']}) "
          f"| into anchored B {r[v]['exact_into_anch_B']:.4f} | share of links into unanchored same-limb B {r[v]['share_links_into_unanch_same']:.4f}")
print(f"   true share of successors that are unanchored same-limb: {np.mean((B_un & same)[h]):.4f}; changed links test_like vs as_trained {np.mean(SU['test_like'][mask] != SU['as_trained'][mask]):.4f}")
json.dump(dict(fit=a.fit, mode=a.mode, subjects=subs, **r), open(os.path.join(HERE_E, f"audit2_{TAG}.json"), "w"), indent=1)
