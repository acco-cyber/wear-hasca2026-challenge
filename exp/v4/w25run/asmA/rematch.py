"""asmA variant R: assembly-informed RE-MATCHING. The nested stage-2 log-odds of every candidate pair (testD
pred_K7_deaug_s2.npy, identical tables rebuilt with tdlib) are shifted by +delta when the pair agrees with the assembly
(j sits at coord(i)+1 in i's group of >= gmin tiles) and by -delta when it contradicts it (i has another assembly successor,
j another assembly predecessor, or j lies in i's group at a different coordinate); then the same Hungarian matching +
7 perturbed members (tau 0.3, seeds as build_test_links) and qnorm. (assembly config, gmin, delta) chosen NESTED by fold
(member-0 exact on the other folds' subjects).
  python rematch.py --grid grid1 --cfgs t0.9_m0.5_c1,t0.5_m0.5_c1 --gmins 2,10 --deltas 1,2,3 --out links_asmA_R.npz"""
import os, sys, time, json, argparse
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__))
TD = r"E:\Claude code\wear\exp\v4\w25run\testD"
sys.path.insert(0, TD)
_argv = sys.argv; sys.argv = [_argv[0]]
from tdlib import load_stage, K7, chain_tuple, topk_cands, subject_table, match, qnorm   # noqa: E402  (read-only import)
sys.argv = _argv

ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="grid1"); ap.add_argument("--cfgs", default="t0.9_m0.5_c1,t0.5_m0.5_c1")
ap.add_argument("--gmins", default="2,10"); ap.add_argument("--deltas", default="1,2,3"); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--out", default="links_asmA_R.npz"); ap.add_argument("--subjects", default=""); ap.add_argument("--crit", default="exact", choices=["exact", "cross"])
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


S = load_stage(K7); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]; fold = S["oof_fold"]; N = len(sbj)
y = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)["oof_y"]
LZP = os.path.join(K7, "link_logodds.npz")
REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]
subs_all = [int(x) for x in np.unique(sbj)]
subs = [int(x) for x in a.subjects.split(",")] if a.subjects else subs_all
PRED = np.load(os.path.join(TD, "cache", "pred_K7_deaug_s2.npy"))
BASE = np.load(os.path.join(TD, "links_oof_K7_deaug.npz"))
_G = np.load(os.path.join(HERE, "cache", f"sim_{a.grid}.npz"))
CFGS = a.cfgs.split(",")
G = {f"{c}_{k}": _G[f"{c}_{k}"] for c in CFGS for k in ("asu", "root", "coord", "gtiles")}
OSTART = np.asarray(S["oof_start"]); OREC = np.asarray(S["oof_rec"]); GMINS = [int(x) for x in a.gmins.split(",")]; DELTAS = [float(x) for x in a.deltas.split(",")]
COMBOS = [(c, g, d) for c in CFGS for g in GMINS for d in DELTAS]


def pairs_of(s):
    """build_test_links.sim_table (mode deaug) -> only what the matching needs"""
    loc = np.flatnonzero(sbj == s); n = len(loc)
    rows = loc[np.lexsort((OSTART[loc], OREC[loc]))]
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    pos = p_of[loc]; lim = sens[loc]
    z = np.load(os.path.join(TD, "cache", f"chaindeaug_s{s}.npz")); assert (z["rows"] == rows).all()
    anch = ~z["aug"][lim, pos]
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
    chains = [chain_tuple(z[f"succ{L}"], z[f"conf{L}"]) for L in range(4)]
    LZ = np.load(LZP); cand, Lo = topk_cands(LZ[f"oof_{s}_cand"], LZ[f"oof_{s}_L"], 50)
    t = subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, tl, 16)
    return s, dict(n=n, pi=t["pi"], pj=t["pj"], loc=loc)


def shifted(t, pr, c, gmin, delta):
    loc = t["loc"]; n = t["n"]; pi, pj = t["pi"], t["pj"]
    asu_g = G[c + "_asu"][loc]; gl = np.full(N, -1); gl[loc] = np.arange(n)
    asu = np.where(asu_g >= 0, gl[np.maximum(asu_g, 0)], -1); root = G[c + "_root"][loc]; coord = G[c + "_coord"][loc]; gt = G[c + "_gtiles"][loc]
    use = (asu >= 0) & (gt >= gmin)
    hp = np.full(n, -1); hp[asu[use]] = np.flatnonzero(use)
    agree = use[pi] & (asu[pi] == pj)
    contra = ~agree & ((use[pi] & (asu[pi] != pj)) | ((hp[pj] >= 0) & (hp[pj] != pi)) | ((gt[pi] >= gmin) & (root[pj] == root[pi]) & (coord[pj] != coord[pi] + 1)))
    return (pr + delta * agree - delta * contra).astype(np.float32)


def member0(s, t, pr):
    out = {}
    _, r = match(t, pr, 1, s); out["base"] = r[0][0]
    for cb in COMBOS:
        _, r = match(t, shifted(t, pr, *cb), 1, s); out[cb] = r[0][0]
    return s, out


def members8(s, t, pr, cb):
    _, res = match(t, shifted(t, pr, *cb) if cb is not None else pr, 8, s)
    return s, res


tabs = dict(Parallel(n_jobs=a.jobs)(delayed(pairs_of)(s) for s in sorted(subs_all, key=lambda s: -(sbj == s).sum())))
log("pair tables rebuilt")
off = 0; prs = {}
for s in subs_all:
    k = len(tabs[s]["pi"]); prs[s] = PRED[off:off + k]; off += k
assert off == len(PRED), (off, len(PRED))
R0 = dict(Parallel(n_jobs=a.jobs)(delayed(member0)(s, tabs[s], prs[s]) for s in sorted(subs, key=lambda s: -(sbj == s).sum())))
log("member-0 matchings done")
mask = np.isin(sbj, subs); h = (ts >= 0) & mask


def to_global(s, su):
    loc = tabs[s]["loc"]; return np.where(su >= 0, loc[np.maximum(su, 0)], -1)


OK = {}; XW = {}; LKD = {}
for key in ["base"] + COMBOS:
    g = np.full(N, -1)
    for s in subs:
        g[tabs[s]["loc"]] = to_global(s, R0[s][key])
    OK[key] = (g == ts) & h
    if key == "base":
        log(f"harness: rebuilt base member 0 == links_oof_K7_deaug member 0: {np.mean(g[mask] == BASE['oof_succ'][0][mask]):.4f}")
    lk = mask & (g >= 0); j = g[lk]
    xw = np.zeros(N, bool); xw[lk] = (y[lk] != y[j]) & (j != ts[lk]); XW[key] = xw; LKD[key] = lk
    log(f"{key}: exact {OK[key][h].mean():.4f} | folds " + " ".join(f"{OK[key][h & (fold == f)].mean():.4f}" for f in range(5))
        + f" | cross-label {np.mean(y[lk] != y[j]):.4f} wrong&cross {np.mean((y[lk] != y[j]) & (j != ts[lk])):.4f}")
choice = {}
for f in range(5):
    tr = h & (fold != f)
    trm = mask & (fold != f)
    if a.crit == "exact":
        best = max(COMBOS, key=lambda cb: OK[cb][tr].sum())
    else:                                                      # share of links that are wrong AND join different labels
        best = min(COMBOS, key=lambda cb: (XW[cb][trm].sum() / LKD[cb][trm].sum(), -OK[cb][tr].sum()))
        log(f"   fold {f}: train wrong&cross best {XW[best][trm].sum() / LKD[best][trm].sum():.5f} base {XW['base'][trm].sum() / LKD['base'][trm].sum():.5f}")
    choice[f] = best
    log(f"fold {f}: chose {best} (train exact {OK[best][tr].mean():.4f} vs base {OK['base'][tr].mean():.4f})")
fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
R8 = dict(Parallel(n_jobs=a.jobs)(delayed(members8)(s, tabs[s], prs[s], choice[fold_of[s]]) for s in sorted(subs, key=lambda s: -(sbj == s).sum())))
Su = BASE["oof_succ"].copy().astype(np.int64); Sc = BASE["oof_score"].copy().astype(np.float32)
for s in subs:
    loc = tabs[s]["loc"]
    for k, (su, sc) in enumerate(R8[s]):
        Su[k, loc] = to_global(s, su); Sc[k, loc] = sc
for k in range(8):
    Sc[k] = qnorm(Su[k], Sc[k], sbj, REF)
e = Su[0][h] == ts[h]; lk = mask & (Su[0] >= 0); j = Su[0][lk]
log(f"NESTED R: exact {e.mean():.4f} (base {OK['base'][h].mean():.4f}) | folds " + " ".join(f"{np.mean(e[fold[h] == f]):.4f}" for f in range(5))
    + f" | perturbed {np.mean([np.mean(Su[k][h] == ts[h]) for k in range(1, 8)]):.4f} | cross-label {np.mean(y[lk] != y[j]):.4f} wrong&cross {np.mean((y[lk] != y[j]) & (j != ts[lk])):.4f}"
    + f" | changed m0 {np.mean(Su[0][mask] != BASE['oof_succ'][0][mask]):.4f}")
LT = np.load(os.path.join(TD, "links_test_K7_deaug.npz"))
np.savez(os.path.join(HERE, a.out), oof_succ=Su, oof_score=Sc, test_succ=LT["test_succ"], test_score=LT["test_score"])
json.dump(dict(choice={int(k): list(map(str, v)) for k, v in choice.items()}, nested_exact=float(e.mean())), open(os.path.join(HERE, a.out[:-4] + "_choice.json"), "w"), indent=1)
log("wrote", a.out)
