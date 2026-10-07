"""asmB on the real test (subjects 22-25): same solver / parameters as the nested OOF choice (thr 0.9, wmin 0.8, ratio 2,
smin 0, l3w 0.3, gmin 10) on the K7 de-augmented chain-rescored test links. Writes links_asmB_test.npz with
test_succ/test_score (assembly) + oof_succ/oof_score (nested OOF links)."""
import os, json
import numpy as np
from asmlib import *
from solver import Solver, build_votes, pair_votes

P = dict(thr=0.9, wmin=0.8, ratio=2.0, smin=0.0, l3w=0.3, l3k=3, chain_w=1.0, gmin=10)
S = stage(); tsbj = S["test_sbj"]; NT = len(tsbj)
LK = np.load(os.path.join(TD, "links_test_K7_deaug.npz")); TSU, TSC = LK["test_succ"], LK["test_score"]
LZ = np.load(os.path.join(K7, "link_logodds.npz"))
REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]; HIGH = float(np.percentile(REF, 99))
nested = np.load(os.path.join(HERE, "links_asmB_nested.npz"))
Su_out = TSU.copy(); Sc_out = TSC.copy(); stats = {}
for s in (22, 23, 24, 25):
    d = load_test(s); n = d["n"]; loc = d["loc"]
    su, sc = local_links(TSU, TSC, loc, NT)
    cand = LZ[f"test_{s}_cand"].astype(np.int64); Lo = LZ[f"test_{s}_L"].astype(np.float32)
    sv = Solver(d, P["thr"])
    fa, fb, dd, w = build_votes(d, sv.frag, sv.off, P["thr"], su, sc, cand, Lo, P["l3w"], P["l3k"], P["chain_w"])
    sv.solve(fa, fb, dd, w, P["wmin"], P["ratio"], P["smin"])
    asu, status, g, c, size = sv.derive(P["gmin"])
    pi, pj, pw = pair_votes(su, sc, cand, Lo, 0.3, 5); o = np.argsort(-pw, kind="stable"); pi, pj = pi[o], pj[o]
    has = asu >= 0; claimed = np.zeros(n, bool); claimed[asu[has]] = True
    nsu = su.copy(); nsc = sc.copy()
    for k in range(su.shape[0]):
        x = su[k]; drop = (~has) & (x >= 0) & claimed[np.maximum(x, 0)]
        nsu[k] = np.where(has, asu, np.where(drop, -1, x)); nsc[k] = np.where(has, HIGH, np.where(drop, -50.0, sc[k]))
        need = np.zeros(n, bool); need[drop] = True; used = np.zeros(n, bool); used[nsu[k][nsu[k] >= 0]] = True
        for i_, j_ in zip(pi[need[pi]], pj[need[pi]]):
            if need[i_] and not used[j_]:
                hit = su[:, i_] == j_
                nsu[k][i_] = j_; nsc[k][i_] = float(sc[hit, i_].max()) if hit.any() else -3.0
                need[i_] = False; used[j_] = True
        v = nsu[k][nsu[k] >= 0]; assert len(np.unique(v)) == len(v)
    Su_out[:, loc] = np.where(nsu >= 0, loc[np.maximum(nsu, 0)], -1); Sc_out[:, loc] = nsc
    ing = (g >= 0) & (size >= P["gmin"])
    stats[s] = dict(n=n, coverage=float(ing.mean()), asm_links=float(has.mean()), asm_agree_base=float(np.mean(asu[has] == su[0][has])),
                    member0_changed=float(np.mean(nsu[0] != su[0])), linked=float(np.mean(nsu[0] >= 0)), linked_base=float(np.mean(su[0] >= 0)),
                    merges=sv.n_merge, collisions=sv.n_coll)
    print(s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats[s].items()}), flush=True)
np.savez(os.path.join(HERE, "links_asmB_test.npz"), test_succ=Su_out, test_score=Sc_out, oof_succ=nested["oof_succ"], oof_score=nested["oof_score"])
json.dump(dict(params=P, stats=stats), open(os.path.join(HERE, "test_info.json"), "w"), indent=1)
print("wrote links_asmB_test.npz")
