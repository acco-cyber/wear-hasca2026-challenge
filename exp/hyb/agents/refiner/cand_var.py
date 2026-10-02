"""Decision variants of the candidate refiner from saved scores (cand_scores.npz): any candidate / only ours / only disagree rows;
session bootstrap vs the hand gate; test change stats + agreement with the LB-best file."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP, macro_f1
OUT = os.path.dirname(os.path.abspath(__file__)); W = r"E:\Claude code\wear"
z = np.load(os.path.join(OUT, "cand_scores.npz")); g = np.load(os.path.join(OUT, "graph_cv.npz")); gt = np.load(os.path.join(OUT, "graph_test.npz"))
sm = np.load(os.path.join(KEEP, "sim_meta.npz")); y = sm["y"].astype(np.int64); fold = sm["fold"]; rec = sm["rec"]
t, o, S, gl = g["base"], g["ours"], g["has"], g["gate_lab"]
ref = pd.read_csv(os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv")).sort_values("id").target_feature.to_numpy()


def decide(n, rid, cc, sc, t, o, delta, mode):
    best = np.full(n, -1.0); arg = t.copy(); st = np.zeros(n)
    m_t = cc == t[rid]; st[rid[m_t]] = sc[m_t]
    if mode == "ours":                              # only the our-label candidate may replace theirs
        m = cc == o[rid]; best[rid[m]] = sc[m]; arg[rid[m]] = cc[m]
    else:
        q = np.argsort(sc, kind="stable"); best[rid[q]] = sc[q]; arg[rid[q]] = cc[q]
    sw = (arg != t) & (best - st > delta)
    if mode == "dis":
        sw &= (o != t)
    lab = t.copy(); lab[sw] = arg[sw]; return lab


grid = np.round(np.arange(-0.05, 0.301, 0.01), 2); res = {}
for mode in ("any", "ours", "dis"):
    f1 = [macro_f1(y[S], decide(len(t), z["rid"], z["cc"], z["sc"], t, o, d, mode)[S]) for d in grid]; db = grid[int(np.argmax(f1))]
    lab = decide(len(t), z["rid"], z["cc"], z["sc"], t, o, db, mode); lab_n = t.copy()
    for f in range(5):
        m = S & (fold != f); d_f = grid[int(np.argmax([macro_f1(y[m], decide(len(t), z["rid"], z["cc"], z["sc"], t, o, d, mode)[m]) for d in grid]))]
        lf = decide(len(t), z["rid"], z["cc"], z["sc"], t, o, d_f, mode); lab_n[fold == f] = lf[fold == f]
    pf = [macro_f1(y[S & (fold == f)], lab[S & (fold == f)]) - macro_f1(y[S & (fold == f)], gl[S & (fold == f)]) for f in range(5)]
    labt = decide(len(gt["base"]), z["trid"], z["tc"], z["st"], gt["base"], gt["ours"], db, mode)
    print(f"{mode:5s} delta={db:.2f}: F1(S) {macro_f1(y[S], lab[S]):.4f} nested {macro_f1(y[S], lab_n[S]):.4f} | minus gate per fold {np.round(pf, 4).tolist()} "
          f"| OOF changed {np.mean(lab[S] != t[S]):.4f} | TEST changed {np.mean(labt != gt['base']):.4f}, agree w/ gl7 {np.mean(labt == ref):.4f}")
    res[mode] = (lab, labt, db)
sess = np.unique(rec[S]); idx = {s: np.flatnonzero(S & (rec == s)) for s in sess}; rng = np.random.default_rng(0)
for mode in ("any", "ours"):
    lab = res[mode][0]; d = []
    for b in range(1000):
        ii = np.concatenate([idx[s] for s in rng.choice(sess, len(sess))]); d.append(macro_f1(y[ii], lab[ii]) - macro_f1(y[ii], gl[ii]))
    d = np.array(d); wins = sum(macro_f1(y[idx[s]], lab[idx[s]]) > macro_f1(y[idx[s]], gl[idx[s]]) for s in sess)
    print(f"bootstrap {mode}: minus gate mean {d.mean():.4f} 90% CI [{np.quantile(d, .05):.4f}, {np.quantile(d, .95):.4f}] P>0 {np.mean(d > 0):.3f}; sessions won {wins}/18")
    np.save(os.path.join(OUT, f"cand_{mode}_test_lab.npy"), res[mode][1])
