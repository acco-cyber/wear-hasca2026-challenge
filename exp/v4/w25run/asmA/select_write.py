"""asmA (d): nested-by-fold choice of (thr, mmin, gmin) among the simulated assembly configs (cmin fixed a priori, drop_contra
fixed a priori), selection criterion = member-0 exact-successor rate on the OTHER folds' subjects; then compose the 8-member
OOF links per fold and write them with K7's chain-rescored TEST links as placeholder.
  python select_write.py --grid grid1 --cmin 2 --out links_asmA_c2.npz"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from asmlib import compose                                     # noqa: E402

W = r"E:\Claude code\wear"
TD = os.path.join(W, "exp", "v4", "w25run", "testD")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="grid1"); ap.add_argument("--cmin", type=int, default=2)
ap.add_argument("--out", default="links_asmA_c2.npz"); ap.add_argument("--gmins", default="2,10,30")
a = ap.parse_args()
S = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
sbj = S["oof_sbj"].astype(np.int64); y = S["oof_y"]; fold = S["oof_fold"].astype(np.int64); ts = S["true_succ"].astype(np.int64); N = len(sbj)
LK = np.load(os.path.join(TD, "links_oof_K7_deaug.npz")); OSU = LK["oof_succ"].astype(np.int64); OSC = LK["oof_score"].astype(np.float32)
LT = np.load(os.path.join(TD, "links_test_K7_deaug.npz"))
REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]; HI = float(np.percentile(REF, 99))
G = np.load(os.path.join(HERE, "cache", f"sim_{a.grid}.npz"))
cfgs = sorted({k.rsplit("_", 1)[0] for k in G.files if k.endswith(f"_c{a.cmin}_asu")})
h = ts >= 0
results = {}
for c in cfgs:
    for gmin in [int(x) for x in a.gmins.split(",")]:
        r = dict(asu=G[c + "_asu"], root=G[c + "_root"], coord=G[c + "_coord"], gtiles=G[c + "_gtiles"])
        SU, SC, use = compose(OSU, OSC, r, gmin, HI, True)
        results[(c, gmin)] = (SU, SC, (SU[0] == ts) & h)
base_ok = (OSU[0] == ts) & h
choice = {}; OUT_SU = OSU.copy(); OUT_SC = OSC.copy()
for f in range(5):
    tr = h & (fold != f)
    best = max(results, key=lambda k: results[k][2][tr].sum() / tr.sum())
    choice[f] = dict(cfg=best[0], gmin=best[1], train_exact=float(results[best][2][tr].sum() / tr.sum()), train_base=float(base_ok[tr].sum() / tr.sum()))
    m = fold == f
    OUT_SU[:, m] = results[best][0][:, m]; OUT_SC[:, m] = results[best][1][:, m]
    print(f"fold {f}: chose {best} train exact {choice[f]['train_exact']:.4f} (base {choice[f]['train_base']:.4f})")
e = OUT_SU[0][h] == ts[h]; lk = OUT_SU[0] >= 0; j = OUT_SU[0][lk]
print(f"NESTED OOF exact {e.mean():.4f} (base {base_ok[h].mean():.4f}); per fold " + " ".join(f"{np.mean(e[fold[h] == f]):.4f}" for f in range(5))
      + f"; cross-label {np.mean(y[lk] != y[j]):.4f} wrong&cross {np.mean((y[lk] != y[j]) & (j != ts[lk])):.4f}; changed links m0 {np.mean(OUT_SU[0] != OSU[0]):.4f}")
for k in range(8):                                             # one-to-one check
    s_ = OUT_SU[k][OUT_SU[k] >= 0]; assert len(np.unique(s_)) == len(s_), k
    assert (sbj[OUT_SU[k][OUT_SU[k] >= 0]] == sbj[OUT_SU[k] >= 0]).all()
np.savez(os.path.join(HERE, a.out), oof_succ=OUT_SU, oof_score=OUT_SC, test_succ=LT["test_succ"], test_score=LT["test_score"])
json.dump(dict(choice={int(k): v for k, v in choice.items()}, nested_exact=float(e.mean())), open(os.path.join(HERE, a.out[:-4] + "_choice.json"), "w"), indent=1)
print("wrote", os.path.join(HERE, a.out))
