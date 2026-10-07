"""asmAudit: reproducibility + anti-leak sensitivity of the asmT (= asmB solver) simulation and test links.
  python audit_repro.py --fit K7|K9
Variants on the 22 training subjects (wmin 0.8 = the nested pick of every fold and the test pick):
  base     : seed 12345 node permutation (exactly asmT's run) -> must equal asmT oof_succ
  seed777  : another random node permutation
  ident    : NO node permutation (node id = true time) -> shows whether the solver could exploit node order
  tshuf    : seed 12345 node permutation + random relabel of the subject-local tile ids (OOF tile order == time order!)
Test: rerun subjects 22-25 at wmin 0.8 -> must equal asmT test_succ. Imports asmB read-only; writes only here."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(R, "asmB"))
from asmlib import stage, load_test, local_links, CACHE, TD, K7, K9   # noqa: E402
from solver import Solver, build_votes, pair_votes                    # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--fit", default="K7"); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--variants", default="base,seed777,ident,tshuf")
a = ap.parse_args()
FIT = {"K7": K7, "K9": K9}[a.fit]
P = dict(thr=0.9, ratio=2.0, smin=0.0, l3w=0.3, l3k=3, chain_w=1.0, gmin=10, wmin=0.8)
S = stage(FIT); sbj = S["oof_sbj"]; N = len(sbj); tsbj = S["test_sbj"]; NT = len(tsbj)
HIGH = float(np.percentile(np.load(os.path.join(FIT, "links.npz"))["qn_ref"], 99))
OL = np.load(os.path.join(TD, f"links_oof_{a.fit}_deaug.npz")); OSU, OSC = OL["oof_succ"], OL["oof_score"]
TL = np.load(os.path.join(TD, f"links_test_{a.fit}_deaug.npz")); TSU, TSC = TL["test_succ"], TL["test_score"]
LZP = os.path.join(FIT, "link_logodds.npz")
ASMT = np.load(os.path.join(R, "asmT", f"links_asm_{a.fit}.npz"))
T0 = time.time(); log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def load_sim_v(s, seed, ident):
    """copy of asmB asmlib.load_sim with an optional identity node permutation (audit only)"""
    loc = np.flatnonzero(sbj == s); n = len(loc)
    z = np.load(os.path.join(CACHE, f"chaindeaug_s{s}.npz")); rows = z["rows"]
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    tpos = p_of[loc]; lim = S["sensor_oof"][loc]; anch = ~z["aug"][lim, tpos]
    rng = np.random.default_rng(seed + 1000 * s)
    perm = [np.arange(n) if ident else rng.permutation(n) for _ in range(4)]
    succ, conf = [], []
    for L in range(4):
        su0 = z[f"succ{L}"].astype(np.int64); cf0 = z[f"conf{L}"].astype(np.float32); Pm = perm[L]
        su = np.full(n, -1, np.int64); cf = np.zeros(n, np.float32)
        su[Pm] = np.where(su0 >= 0, Pm[np.maximum(su0, 0)], -1); cf[Pm] = cf0; succ.append(su); conf.append(cf)
    pos = np.array([perm[lim[i]][tpos[i]] for i in range(n)], np.int64); pos = np.where(anch, pos, 0)
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    ts = S["true_succ"][loc]; tl = np.where(ts >= 0, l_of[np.maximum(ts, 0)], -1)
    d = dict(s=s, n=n, loc=loc, lim=lim, anch=anch, pos=pos, owner=owner, succ=succ, conf=conf, fold=int(S["oof_fold"][loc[0]]))
    return d, dict(tl=tl, y=S["oof_y"][loc])


def final_links(su, sc, asu, pv, high):
    """verbatim logic of asmT run_asmT.final_links"""
    M, n = su.shape; has = asu >= 0; claimed = np.zeros(n, bool); claimed[asu[has]] = True
    nsu = su.copy(); nsc = sc.copy()
    pi, pj, pw = pv; o = np.argsort(-pw, kind="stable"); pi, pj = pi[o], pj[o]
    for k in range(M):
        x = su[k]; drop = (~has) & (x >= 0) & claimed[np.maximum(x, 0)]
        nsu[k] = np.where(has, asu, np.where(drop, -1, x)); nsc[k] = np.where(has, high, np.where(drop, -50.0, sc[k]))
        need = np.zeros(n, bool); need[drop] = True; used = np.zeros(n, bool); used[nsu[k][nsu[k] >= 0]] = True
        for i_, j_ in zip(pi[need[pi]], pj[need[pi]]):
            if need[i_] and not used[j_]:
                hit = su[:, i_] == j_
                nsu[k][i_] = j_; nsc[k][i_] = float(sc[hit, i_].max()) if hit.any() else -3.0
                need[i_] = False; used[j_] = True
        v = nsu[k][nsu[k] >= 0]; assert len(np.unique(v)) == len(v)
    return nsu, nsc.astype(np.float32)


def solve(d, su, sc, cand, Lo):
    sv0 = Solver(d, P["thr"])
    fa, fb, dd, w = build_votes(d, sv0.frag, sv0.off, P["thr"], su, sc, cand, Lo, P["l3w"], P["l3k"], P["chain_w"])
    pv = pair_votes(su, sc, cand, Lo, 0.3, 5)
    sv = Solver(d, P["thr"]); sv.solve(fa, fb, dd, w, P["wmin"], P["ratio"], P["smin"])
    asu, status, g, c, size = sv.derive(P["gmin"])
    nsu, nsc = final_links(su, sc, asu, pv, HIGH)
    return nsu, nsc, asu, (g >= 0) & (size >= P["gmin"])


def relabel(d, su, sc, cand, Lo, q):
    """tile i -> new id q[i] everywhere"""
    qi = np.argsort(q); mp = lambda x: np.where(x >= 0, q[np.maximum(x, 0)], -1)
    d2 = dict(d); d2["lim"] = d["lim"][qi]; d2["anch"] = d["anch"][qi]; d2["pos"] = d["pos"][qi]; d2["owner"] = mp(d["owner"])
    return d2, mp(su[:, qi]), sc[:, qi], mp(cand[qi]), Lo[qi], qi


def run_sim(s, var):
    seed = 777 if var == "seed777" else 12345
    d, tr = load_sim_v(s, seed, var == "ident")
    su, sc = local_links(OSU, OSC, d["loc"], N)
    LZ = np.load(LZP); cand = LZ[f"oof_{s}_cand"].astype(np.int64); Lo = LZ[f"oof_{s}_L"].astype(np.float32)
    if var == "tshuf":
        q = np.random.default_rng(99 + s).permutation(d["n"])
        d2, su2, sc2, c2, L2, qi = relabel(d, su, sc, cand, Lo, q)
        nsu2, nsc2, asu2, ing2 = solve(d2, su2, sc2, c2, L2)
        back = lambda x: np.where(x >= 0, qi[np.maximum(x, 0)], -1)
        nsu, nsc, asu, ing = back(nsu2[:, q]), nsc2[:, q], back(asu2[q]), ing2[q]
    else:
        nsu, nsc, asu, ing = solve(d, su, sc, cand, Lo)
    return dict(s=s, loc=d["loc"], su=su, nsu=nsu, nsc=nsc, asu=asu, ing=ing, tl=tr["tl"], y=tr["y"])


def score(rs):
    e, eb, x, xb, lk, lkb, ch, cov, ae = [], [], [], [], [], [], [], [], []
    for r in rs:
        tl, y = r["tl"], r["y"]; h = tl >= 0; b0, n0 = r["su"][0], r["nsu"][0]
        e.append(n0[h] == tl[h]); eb.append(b0[h] == tl[h]); ch.append(n0 != b0); cov.append(r["ing"])
        x.append(np.where(n0 >= 0, (y != y[np.maximum(n0, 0)]) & (n0 != tl), False)); lk.append(n0 >= 0)
        xb.append(np.where(b0 >= 0, (y != y[np.maximum(b0, 0)]) & (b0 != tl), False)); lkb.append(b0 >= 0)
        m = (r["asu"] >= 0) & h; ae.append(r["asu"][m] == tl[m])
    C = lambda v: np.concatenate(v)
    return dict(exact_new=float(C(e).mean()), exact_base=float(C(eb).mean()), cross_new=float(C(x).sum() / C(lk).sum()),
                cross_base=float(C(xb).sum() / C(lkb).sum()), m0_changed=float(C(ch).mean()), coverage_ge10=float(C(cov).mean()),
                asm_link_exact=float(C(ae).mean()), asm_links=int(len(C(ae))))


out = dict(fit=a.fit, params=P, high=HIGH, sim={})
subs = [int(v) for v in np.unique(sbj)]
for var in a.variants.split(","):
    rs = Parallel(n_jobs=a.jobs)(delayed(run_sim)(s, var) for s in sorted(subs, key=lambda s: -(sbj == s).sum()))
    sc_ = score(rs)
    Su = np.full((8, N), -1, np.int64); Sc = np.full((8, N), -50.0, np.float32)
    for r in rs:
        Su[:, r["loc"]] = np.where(r["nsu"] >= 0, r["loc"][np.maximum(r["nsu"], 0)], -1); Sc[:, r["loc"]] = r["nsc"]
    sc_["eq_asmT_oof_succ"] = bool((Su == ASMT["oof_succ"]).all()); sc_["share_eq_asmT_oof_succ"] = float(np.mean(Su == ASMT["oof_succ"]))
    sc_["eq_asmT_oof_score"] = bool(np.allclose(Sc, ASMT["oof_score"]))
    out["sim"][var] = sc_; log(f"{a.fit} {var}: {json.dumps(sc_)}")
    if var == "tshuf":
        np.savez(os.path.join(HERE, f"oof_tshuf_{a.fit}.npz"), oof_succ=Su, oof_score=Sc)
# test reproduction
TSu = TSU.astype(np.int64).copy(); TSc = TSC.astype(np.float32).copy()
for s in (22, 23, 24, 25):
    d = load_test(s, FIT); su, sc = local_links(TSU, TSC, d["loc"], NT)
    LZ = np.load(LZP); cand = LZ[f"test_{s}_cand"].astype(np.int64); Lo = LZ[f"test_{s}_L"].astype(np.float32)
    nsu, nsc, asu, ing = solve(d, su, sc, cand, Lo); loc = d["loc"]
    TSu[:, loc] = np.where(nsu >= 0, loc[np.maximum(nsu, 0)], -1); TSc[:, loc] = nsc
out["test_eq_asmT_succ"] = bool((TSu == ASMT["test_succ"]).all()); out["test_eq_asmT_score"] = bool(np.allclose(TSc, ASMT["test_score"]))
log(f"test reproduction: succ equal {out['test_eq_asmT_succ']} score equal {out['test_eq_asmT_score']}")
json.dump(out, open(os.path.join(HERE, f"audit_repro_{a.fit}.json"), "w"), indent=1)
