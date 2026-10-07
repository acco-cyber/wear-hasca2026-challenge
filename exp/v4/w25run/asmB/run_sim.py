"""asmB: run the global offset solver on the permuted-node simulation of the training subjects and score it.
  python run_sim.py --subs 5,19 --wmin 0.8 --ratio 2 --thr 0.9 --l3w 0.3 [--save tag]"""
import os, sys, time, argparse, json
import numpy as np
from joblib import Parallel, delayed
from asmlib import *
from solver import Solver, build_votes, pair_votes

ap = argparse.ArgumentParser()
ap.add_argument("--subs", default="all"); ap.add_argument("--wmin", type=float, default=0.8); ap.add_argument("--ratio", type=float, default=2.0)
ap.add_argument("--thr", type=float, default=0.9); ap.add_argument("--l3w", type=float, default=0.3); ap.add_argument("--l3k", type=int, default=3)
ap.add_argument("--chain_w", type=float, default=1.0); ap.add_argument("--gmin", type=int, default=10)
ap.add_argument("--links", default=os.path.join(TD, "links_oof_K7_deaug.npz")); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--smin", type=float, default=-99.0); ap.add_argument("--relink", type=int, default=1); ap.add_argument("--save", default=""); ap.add_argument("--seed", type=int, default=12345)
a = ap.parse_args()
T0 = time.time()
S = stage(); sbj = S["oof_sbj"]; N = len(sbj)
subs = [int(x) for x in np.unique(sbj)] if a.subs == "all" else [int(x) for x in a.subs.split(",")]
LK = np.load(a.links); SU, SC = LK["oof_succ"], LK["oof_score"]
LZP = os.path.join(K7, "link_logodds.npz")


def run(s):
    t0 = time.time()
    d = load_sim(s, a.seed); n = d["n"]; tr = d["truth"]
    su, sc = local_links(SU, SC, d["loc"], N)
    LZ = np.load(LZP); cand = LZ[f"oof_{s}_cand"].astype(np.int64); Lo = LZ[f"oof_{s}_L"].astype(np.float32)
    sv = Solver(d, a.thr)
    fa, fb, dd, w = build_votes(d, sv.frag, sv.off, a.thr, su, sc, cand, Lo, a.l3w, a.l3k, a.chain_w)
    sv.solve(fa, fb, dd, w, a.wmin, a.ratio, a.smin)
    pv = pair_votes(su, sc, cand, Lo, 0.3, 5)
    asu, status, g, c, size = sv.derive(a.gmin)
    # group purity (scoring only): share of tiles whose (true position - coordinate) equals their group's mode
    pure = np.zeros(n, bool)
    ing = np.flatnonzero((g >= 0) & (size >= a.gmin))
    if len(ing):
        sh = tr["tpos"][ing] - c[ing]; key = g[ing]
        for r in np.unique(key):
            m = key == r; v, k = np.unique(sh[m], return_counts=True); pure[ing[m]] = sh[m] == v[np.argmax(k)]
    return dict(s=s, n=n, su=su, sc=sc, asu=asu, status=status, g=g, c=c, size=size, pure=pure, tl=tr["tl"], y=tr["y"],
                anch=d["anch"], loc=d["loc"], fold=d["fold"], nm=sv.n_merge, nc=sv.n_coll, ns=sv.n_srej, pv=pv, F=sv.F, t=time.time() - t0)


def final_links(r, high):
    """8 members: assembly successor (all members, score high) where status 1, else member k of the old links; old links
    into tiles claimed by an assembly link are dropped"""
    su, sc, asu = r["su"], r["sc"], r["asu"]; M = su.shape[0]; n = r["n"]
    has = asu >= 0; claimed = np.zeros(n, bool); claimed[asu[has]] = True
    nsu = su.copy(); nsc = sc.copy()
    pi, pj, pw = r["pv"]; o = np.argsort(-pw, kind="stable"); pi, pj, pw = pi[o], pj[o], pw[o]
    for k in range(M):
        x = su[k]; drop = (~has) & (x >= 0) & claimed[np.maximum(x, 0)]
        nsu[k] = np.where(has, asu, np.where(drop, -1, x)); nsc[k] = np.where(has, high, np.where(drop, -50.0, sc[k]))
        if a.relink:                       # dropped sources: best remaining candidate (votes) whose target is still free
            need = np.zeros(n, bool); need[drop] = True
            used = np.zeros(n, bool); used[nsu[k][nsu[k] >= 0]] = True
            for i_, j_ in zip(pi[need[pi]], pj[need[pi]]):
                if need[i_] and not used[j_]:
                    hit = su[:, i_] == j_
                    nsu[k][i_] = j_; nsc[k][i_] = float(sc[hit, i_].max()) if hit.any() else -3.0
                    need[i_] = False; used[j_] = True
    return nsu, nsc


REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]; HIGH = float(np.percentile(REF, 99))
res = Parallel(n_jobs=a.jobs)(delayed(run)(s) for s in sorted(subs, key=lambda s: -(sbj == s).sum()))
res = sorted(res, key=lambda r: r["s"])
print(f"[{time.time()-T0:.0f}s] params {vars(a)}", flush=True)
agg = {k: [] for k in ["base_e", "new_e", "ing", "h", "cov", "st", "pure", "base_x", "new_x", "true_x", "lk_b", "lk_n", "fold", "asm_e", "asm_b"]}
for r in res:
    nsu, nsc = final_links(r, HIGH)
    tl, y = r["tl"], r["y"]; h = tl >= 0
    b0, n0 = r["su"][0], nsu[0]
    ing = (r["g"] >= 0) & (r["size"] >= a.gmin)
    agg["base_e"].append(b0[h] == tl[h]); agg["new_e"].append(n0[h] == tl[h]); agg["ing"].append(ing[h]); agg["h"].append(h)
    agg["cov"].append(ing); agg["st"].append(r["status"]); agg["pure"].append(r["pure"][ing]); agg["fold"].append(np.full(h.sum(), r["fold"]))
    agg["base_x"].append(np.where(b0 >= 0, (y != y[np.maximum(b0, 0)]) & (b0 != tl), False)); agg["lk_b"].append(b0 >= 0)
    agg["new_x"].append(np.where(n0 >= 0, (y != y[np.maximum(n0, 0)]) & (n0 != tl), False)); agg["lk_n"].append(n0 >= 0)
    agg["true_x"].append(y[h] != y[tl[h]])
    st1 = r["status"] == 1
    agg["asm_e"].append(r["asu"][st1 & h] == tl[st1 & h]); agg["asm_b"].append(b0[st1 & h] == tl[st1 & h])
    print(f"  sbj {r['s']:2d} n {r['n']} F {r['F']} merges {r['nm']} coll {r['nc']} | cov {ing.mean():.3f} pure {r['pure'][ing].mean() if ing.any() else 0:.3f} "
          f"| asm links {st1.mean():.3f} exact {np.mean(r['asu'][st1 & h] == tl[st1 & h]):.3f} (base on same {np.mean(b0[st1 & h] == tl[st1 & h]):.3f}) "
          f"| member0 exact {np.mean(n0[h] == tl[h]):.4f} vs {np.mean(b0[h] == tl[h]):.4f} [{r['t']:.0f}s]", flush=True)
C = {k: np.concatenate(v) for k, v in agg.items()}
st = C["st"]
out = dict(coverage=float(C["cov"].mean()), purity=float(C["pure"].mean()),
           status_share={k: float(np.mean(st == k)) for k in range(5)},
           asm_link_exact=float(C["asm_e"].mean()), base_on_asm=float(C["asm_b"].mean()),
           exact_base=float(C["base_e"].mean()), exact_new=float(C["new_e"].mean()),
           exact_in_groups_base=float(C["base_e"][C["ing"]].mean()), exact_in_groups_new=float(C["new_e"][C["ing"]].mean()),
           exact_out_base=float(C["base_e"][~C["ing"]].mean()), exact_out_new=float(C["new_e"][~C["ing"]].mean()),
           cross_label_base=float(C["base_x"].sum() / C["lk_b"].sum()), cross_label_new=float(C["new_x"].sum() / C["lk_n"].sum()),
           cross_label_true=float(C["true_x"].mean()), linked_base=float(C["lk_b"].mean()), linked_new=float(C["lk_n"].mean()),
           exact_by_fold_base=[float(C["base_e"][C["fold"] == f].mean()) for f in range(5)],
           exact_by_fold_new=[float(C["new_e"][C["fold"] == f].mean()) for f in range(5)])
print(f"[{time.time()-T0:.0f}s] SUMMARY", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
if a.save:
    Su = np.full((8, N), -1, np.int64); Sc = np.full((8, N), -50.0, np.float32)
    for r in res:
        nsu, nsc = final_links(r, HIGH); loc = r["loc"]
        Su[:, loc] = np.where(nsu >= 0, loc[np.maximum(nsu, 0)], -1); Sc[:, loc] = nsc
    np.savez(os.path.join(HERE, f"oof_links_{a.save}.npz"), oof_succ=Su, oof_score=Sc)
    json.dump(out, open(os.path.join(HERE, f"sim_{a.save}.json"), "w"), indent=1)
    print("saved", a.save)
