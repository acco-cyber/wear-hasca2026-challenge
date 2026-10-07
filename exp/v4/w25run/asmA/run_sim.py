"""asmA simulation: greedy vote assembly on all 22 training subjects (deaug test-like anchoring, permuted node + tile ids).
  python run_sim.py --thr 0.5,0.8 --mmin 0.5,1,2 --tag grid1
-> cache/sim_<tag>.npz (per config global asu/root/coord/gtiles), log lines with metrics (a)-(c)."""
import os, sys, time, json, argparse
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from asmlib import build_votes, Assembler, compose            # noqa: E402

W = r"E:\Claude code\wear"
TD = os.path.join(W, "exp", "v4", "w25run", "testD")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--thr", default="0.5,0.8"); ap.add_argument("--mmin", default="0.5,1,2"); ap.add_argument("--cmin", default="2")
ap.add_argument("--lam", type=float, default=0.25); ap.add_argument("--passes", type=int, default=12); ap.add_argument("--nogap", action="store_true")
ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--subjects", default=""); ap.add_argument("--tag", default="grid1")
ap.add_argument("--gmins", default="2,10")
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


S = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
sbj = S["oof_sbj"].astype(np.int64); y = S["oof_y"].astype(np.int64); fold = S["oof_fold"].astype(np.int64)
ts = S["true_succ"].astype(np.int64); sens = S["sensor_oof"].astype(np.int64); N = len(sbj)
LK = np.load(os.path.join(TD, "links_oof_K7_deaug.npz")); OSU = LK["oof_succ"].astype(np.int64); OSC = LK["oof_score"].astype(np.float32)
LZP = os.path.join(K7, "link_logodds.npz")
REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]; HI = float(np.percentile(REF, 99))
subs = [int(x) for x in a.subjects.split(",")] if a.subjects else [int(x) for x in np.unique(sbj)]
CFGS = [(float(t), float(m), int(c)) for t in a.thr.split(",") for m in a.mmin.split(",") for c in a.cmin.split(",")]
GMINS = [int(g) for g in a.gmins.split(",")]


def run_subject(s):
    t0 = time.time()
    loc = np.flatnonzero(sbj == s); n = len(loc); l_of = np.full(N, -1, np.int64); l_of[loc] = np.arange(n)
    z = np.load(os.path.join(TD, "cache", f"chaindeaug_s{s}.npz"))
    rows = z["rows"]; p_of = np.full(N, -1, np.int64); p_of[rows] = np.arange(n); pos = p_of[loc]; lim = sens[loc]
    assert (pos >= 0).all()
    anch = ~z["aug"][lim, pos]                                   # test-like anchoring (right_arm type-ii unanchored)
    rng = np.random.default_rng(7000 + s)
    P = np.stack([rng.permutation(n) for _ in range(4)])         # ANTI-LEAK: node ids of each chain permuted
    succ, conf = [], []
    for L in range(4):
        su = z[f"succ{L}"].astype(np.int64); cf = z[f"conf{L}"].astype(np.float32)
        s2 = np.full(n, -1, np.int64); s2[P[L]] = np.where(su >= 0, P[L][np.maximum(su, 0)], -1)
        c2 = np.zeros(n, np.float32); c2[P[L]] = cf
        succ.append(s2); conf.append(c2)
    posP = np.where(anch, P[lim, pos], -1)
    tp = rng.permutation(n); inv = np.argsort(tp)                # tile ids permuted as well (no tie-breaking by OOF order)
    lim_n = np.empty(n, np.int64); lim_n[tp] = lim; anch_n = np.empty(n, bool); anch_n[tp] = anch; pos_n = np.empty(n, np.int64); pos_n[tp] = posP
    msu = np.where(OSU[:, loc] >= 0, l_of[np.maximum(OSU[:, loc], 0)], -1); msc = OSC[:, loc]
    LZ = np.load(LZP); cand = LZ[f"oof_{s}_cand"].astype(np.int64); Lc = LZ[f"oof_{s}_L"].astype(np.float32)
    vi, vj, vw = build_votes(n, msu, msc, cand, Lc, a.lam)
    out = {}
    for thr, mmin, cmin in CFGS:
        A = Assembler(n, lim_n, anch_n, pos_n, succ, conf, thr)
        hist = A.run(tp[vi], tp[vj], vw, mmin, cmin, a.passes, not a.nogap)
        r = A.result()
        ra = r["asu"][tp]
        out[(thr, mmin, cmin)] = dict(asu=np.where(ra >= 0, inv[np.maximum(ra, 0)], -1), root=r["root"][tp], coord=r["coord"][tp],
                                gtiles=r["gtiles"][tp], gfrags=r["gfrags"][tp], hist=hist, nchain=A.n_chain_frag)
    return s, loc, msu, msc, out, time.time() - t0


res = Parallel(n_jobs=a.jobs)(delayed(run_subject)(s) for s in sorted(subs, key=lambda s: -(sbj == s).sum()))
log(f"subjects done: {len(res)}; per-subject time max {max(r[-1] for r in res):.0f}s")
mask = np.isin(sbj, subs); h = (ts >= 0) & mask


def metrics(su0, extra=None):
    e = su0[h] == ts[h]; lk = mask & (su0 >= 0); j = su0[lk]
    cross = y[lk] != y[j]; wrong = j != ts[lk]
    d = dict(exact=float(e.mean()), linked=float(lk.sum() / mask.sum()), cross_label=float(cross.mean()),
             cross_label_wrong=float((cross & wrong).mean()), fold_exact=[float(np.mean(e[fold[h] == f])) for f in range(5)])
    return d


base = metrics(OSU[0])
log("BASELINE chain-rescored links:", json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in base.items()}))
save = {}
summary = {"baseline": base}
for cfg in CFGS:
    G = {k: np.full(N, -1, np.int64) for k in ("asu", "root", "coord", "gtiles", "gfrags")}
    SUall = np.full((8, N), -1, np.int64); hist_merges = 0
    for s, loc, msu, msc, out, _ in res:
        r = out[cfg]
        G["asu"][loc] = np.where(r["asu"] >= 0, loc[np.maximum(r["asu"], 0)], -1)
        G["root"][loc] = r["root"] + 100000 * s; G["coord"][loc] = r["coord"]; G["gtiles"][loc] = r["gtiles"]; G["gfrags"][loc] = r["gfrags"]
        hist_merges += sum(x[2] for x in r["hist"])
    tag = f"t{cfg[0]}_m{cfg[1]}_c{cfg[2]}"
    for k, v in G.items():
        save[f"{tag}_{k}"] = v
    # assembly link precision by group size
    au = (G["asu"] >= 0) & h
    prec = {}
    for lo_, hi_ in ((1, 2), (2, 10), (10, 50), (50, 200), (200, 10 ** 9)):
        m = au & (G["gtiles"] >= lo_) & (G["gtiles"] < hi_)
        prec[f"{lo_}-{hi_}"] = (int(m.sum()), round(float(np.mean(G["asu"][m] == ts[m])) if m.any() else -1, 4))
    cov10 = float(np.mean(G["gtiles"][mask] >= 10)); cov50 = float(np.mean(G["gtiles"][mask] >= 50))
    log(f"CFG thr {cfg[0]} mmin {cfg[1]} cmin {cfg[2]}: merges {hist_merges}; coverage >=10 {cov10:.4f} >=50 {cov50:.4f}; asm-link share {np.mean(G['asu'][mask] >= 0):.4f}; "
        f"asm precision by group size {prec}")
    for gmin in GMINS:
        lj = (G["asu"] >= 0) & (G["gtiles"] >= gmin)
        for dc in (True, False):
            # compose subject by subject in global ids (compose is id-agnostic)
            SU = OSU.copy()
            res_g = dict(asu=G["asu"], root=G["root"], coord=G["coord"], gtiles=G["gtiles"])
            SU, _, use = compose(OSU, OSC, res_g, gmin, HI, dc)
            mt = metrics(SU[0])
            ing = h & (G["gtiles"] >= 10)
            mt["exact_in_groups10"] = float(np.mean(SU[0][ing] == ts[ing])); mt["base_exact_in_groups10"] = float(np.mean(OSU[0][ing] == ts[ing]))
            mt["asm_links"] = float(use[mask].mean()); mt["asm_prec"] = float(np.mean(G["asu"][use & h] == ts[use & h]))
            uh = use & h; ar = G["asu"][uh] == ts[uh]; br = OSU[0][uh] == ts[uh]
            mt["asm_base_prec"] = float(br.mean()); mt["gain"] = int(np.sum(ar & ~br)); mt["loss"] = int(np.sum(~ar & br))
            mt["changed"] = int(np.sum(G["asu"][uh] != OSU[0][uh]))
            mt["perturbed_exact"] = float(np.mean([np.mean(SU[k][h] == ts[h]) for k in range(1, 8)]))
            summary[f"{tag}_g{gmin}_dc{int(dc)}"] = dict(coverage10=cov10, **mt)
            log(f"   gmin {gmin} drop_contra {int(dc)}: exact {mt['exact']:.4f} (base {base['exact']:.4f}) | in groups>=10 {mt['exact_in_groups10']:.4f} (base {mt['base_exact_in_groups10']:.4f}) "
                f"| asm links {mt['asm_links']:.4f} prec {mt['asm_prec']:.4f} (base there {mt['asm_base_prec']:.4f}, changed {mt['changed']} gain {mt['gain']} loss {mt['loss']}) | linked {mt['linked']:.4f} | cross-label {mt['cross_label']:.4f} (base {base['cross_label']:.4f}) "
                f"wrong&cross {mt['cross_label_wrong']:.4f} (base {base['cross_label_wrong']:.4f}) | perturbed exact {mt['perturbed_exact']:.4f} | folds "
                + " ".join(f"{x:.4f}" for x in mt["fold_exact"]))
np.savez_compressed(os.path.join(CACHE, f"sim_{a.tag}.npz"), **save)
json.dump(dict(args=vars(a), summary=summary), open(os.path.join(HERE, f"sim_{a.tag}.json"), "w"), indent=1)
log("done")
