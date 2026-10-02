"""Binary refiner: on rows where their graph label t and our label o disagree, predict 'take ours'.
python train.py [--fs full|robust|min] [--nocls] [--rounds 300] [--top2] [--test out.csv] [--seeds 3]"""
import os, sys, argparse, json
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1, N_CLS
W = r"E:\Claude code\wear"; OUT = os.path.dirname(os.path.abspath(__file__))
RAWPROB = ["q1", "q2", "q3", "margin", "entQ", "Q_o", "Q_t", "lrQ", "lrP"]
SETS = {"full": None, "robust": RAWPROB,
        "min": RAWPROB + ["lrP0", "lrW", "S_n", "K_sim"],
        "safe": RAWPROB + ["S_dis", "S_n"],
        "noW": RAWPROB + ["S_dis", "S_n", "lrP0", "lrW", "lrP0_pct", "lrW_pct", "rkP0_o", "rkP0_t", "rkW_o", "rkW_t", "rkP_o", "rkP_t"],
        "noL": RAWPROB + ["S_dis", "S_n"] + [f"L_{a}{b}" for a in "bu" for b in "to"] + ["L_n", "L1_bo"]}

ap = argparse.ArgumentParser(); ap.add_argument("--fs", default="full"); ap.add_argument("--nocls", action="store_true")
ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--leaves", type=int, default=15); ap.add_argument("--mdl", type=int, default=50)
ap.add_argument("--top2", action="store_true", help="only rows whose our label is their top-2 (rkQ_o<=1)")
ap.add_argument("--test", default=""); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--tag", default="")
ap.add_argument("--save_oof", default="")
a = ap.parse_args()

cv = np.load(os.path.join(OUT, "feats_cv.npz")); te = np.load(os.path.join(OUT, "feats_test.npz"))
names = [str(x) for x in cv["names"]]
drop = set(SETS[a.fs] or []) | ({"cls_t", "cls_o"} if a.nocls else set())
keep = [i for i, nm in enumerate(names) if nm not in drop]; fn = [names[i] for i in keep]
X, Xt = cv["X"][:, keep], te["X"][:, keep]
y, t, o, has, fold, gl = cv["y"], cv["t"], cv["o"], cv["has"], cv["fold"], cv["gate_lab"]
rk = cv["X"][:, names.index("rkQ_o")]; rkt = te["X"][:, names.index("rkQ_o")]
D = has & (t != o); Dt = te["has"] & (te["t"] != te["o"])
if a.top2:
    D &= rk <= 1; Dt &= rkt <= 1
lab_rows = D & ((y == o) | (y == t)); target = (y == o).astype(np.float32)
cat = [fn.index(c) for c in ("cls_t", "cls_o") if c in fn]
params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.mdl, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=4, verbose=-1, max_bin=63,
              cat_smooth=20, min_data_per_group=50, max_cat_to_onehot=4)
S = has
print(f"fs={a.fs} nocls={a.nocls} top2={a.top2} feats={len(fn)} | disagree rows OOF {D.sum()} ({D[S].mean():.4f} of S), "
      f"labelled {lab_rows.sum()} (ours right {int(target[lab_rows].sum())}, theirs right {int((1 - target[lab_rows]).sum())}); "
      f"neither right {int((D & (y != o) & (y != t)).sum())}; test disagree {Dt.sum()} ({Dt.mean():.4f})", flush=True)

p = np.full(len(y), np.nan); imp = np.zeros(len(fn))
for f in range(5):
    tr = lab_rows & (fold != f); va = D & (fold == f)
    if va.sum() == 0:
        continue
    pp = np.zeros(va.sum())
    for sd in range(a.seeds):
        bst = lgb.train(dict(params, seed=sd), lgb.Dataset(X[tr], target[tr], categorical_feature=cat, free_raw_data=False), a.rounds)
        pp += bst.predict(X[va]) / a.seeds; imp += bst.feature_importance("gain")
    p[va] = pp
from sklearn.metrics import roc_auc_score
print(f"OOF AUC on labelled disagree rows {roc_auc_score(target[lab_rows], p[lab_rows]):.4f}")


def relabel(thr, pv=p, rows=D, base=t, ours=o):
    lab = base.copy(); sw = rows & (pv > thr); lab[sw] = ours[sw]; return lab, sw


def f1S(lab, m=None):
    m = S if m is None else (S & m); return macro_f1(y[m], lab[m])


grid = np.round(np.arange(0.10, 0.951, 0.01), 2)
scores = np.array([f1S(relabel(th)[0]) for th in grid]); best = grid[scores.argmax()]
# nested: threshold chosen on the other folds' OOF rows
lab_nest = t.copy()
for f in range(5):
    m_out = fold != f
    sc = [f1S(relabel(th)[0], m_out) for th in grid]; th_f = grid[int(np.argmax(sc))]
    sw = D & (fold == f) & (p > th_f); lab_nest[sw] = o[sw]
lab_best, sw_best = relabel(best)
print(f"threshold curve: " + " ".join(f"{th:.2f}:{s:.4f}" for th, s in zip(grid[::5], scores[::5])))
rows = [("their labels (no gate)", t), ("hand gate 0.55 top2", gl), (f"refiner thr={best:.2f} (pooled)", lab_best),
        ("refiner nested thr", lab_nest), ("refiner thr=0.50", relabel(0.5)[0])]
print(f"\n{'method':34s} F1(S)   " + " ".join(f"fold{f}  " for f in range(5)) + " switched(S)")
for nm, lab in rows:
    pf = [f1S(lab, fold == f) for f in range(5)]
    print(f"{nm:34s} {f1S(lab):.4f}  " + " ".join(f"{v:.4f}" for v in pf) + f"  {np.mean(lab[S] != t[S]):.4f}")
g_f = np.array([f1S(lab_best, fold == f) - f1S(gl, fold == f) for f in range(5)])
print(f"refiner(pooled) - gate per fold: {np.round(g_f, 4).tolist()} ; folds improved {int((g_f > 0).sum())}/5")
sw_acc = np.mean(y[sw_best] == o[sw_best]); print(f"switched rows {sw_best.sum()}: ours correct {sw_acc:.3f}, theirs correct {np.mean(y[sw_best] == t[sw_best]):.3f}")
gs = S & (gl != t); print(f"hand gate switched rows {gs.sum()}: ours correct {np.mean(y[gs] == gl[gs]):.3f}, theirs correct {np.mean(y[gs] == t[gs]):.3f}")
order = np.argsort(-imp); print("importance (gain) top 15: " + ", ".join(f"{fn[i]}={imp[i] / imp.sum():.3f}" for i in order[:15]))
if a.save_oof:
    np.savez(os.path.join(OUT, a.save_oof), p=p, best=best, lab=lab_best, lab_nest=lab_nest)
if a.test:
    pt = np.zeros(len(Xt))
    for sd in range(a.seeds):
        bst = lgb.train(dict(params, seed=sd), lgb.Dataset(X[lab_rows], target[lab_rows], categorical_feature=cat), a.rounds)
        pt += bst.predict(Xt) / a.seeds
    labt = te["t"].copy(); swt = Dt & (pt > best); labt[swt] = te["o"][swt]
    ref = pd.read_csv(os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv")).sort_values("id").target_feature.to_numpy()
    print(f"\nTEST: disagree {Dt.mean():.4f}; switched {swt.mean():.4f} of rows ({swt.sum()}) = {swt.sum() / max(Dt.sum(), 1):.3f} of disagree "
          f"(OOF: {sw_best[S].mean():.4f} of rows, {sw_best.sum() / D.sum():.3f} of disagree); hand gate switched {np.mean(ref != te['t']):.4f}")
    print(f"agreement with sub_gl7_xl_p03c03_top2_055 (LB 0.91289): {np.mean(labt == ref):.5f}; with their base {np.mean(labt == te['t']):.4f}; "
          f"with ours {np.mean(labt == te['o']):.4f}")
    print(f"test p quantiles on disagree rows: {np.round(np.quantile(pt[Dt], [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist()} ; "
          f"OOF: {np.round(np.quantile(p[D], [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist()}")
    ids = np.arange(len(labt))
    pd.DataFrame({"id": ids, "target_feature": labt.astype(int)}).to_csv(os.path.join(OUT, a.test), index=False)
    np.save(os.path.join(OUT, a.test.replace(".csv", "_p.npy")), pt); print("wrote", os.path.join(OUT, a.test))
