"""(1) hand gate tuned on this OOF (tau grid, top2/plain rules), (2) session-level paired bootstrap refiner vs gate,
(3) OOF-vs-test shift of the window-evidence features on disagree rows.  python eval_extra.py <oof_save.npz>"""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1, KEEP
import graph_lab as G
from sklearn.metrics import roc_auc_score
OUT = os.path.dirname(os.path.abspath(__file__))
g = np.load(os.path.join(OUT, "graph_cv.npz")); cv = np.load(os.path.join(OUT, "feats_cv.npz")); te = np.load(os.path.join(OUT, "feats_test.npz"))
sm = np.load(os.path.join(KEEP, "sim_meta.npz")); y = sm["y"].astype(np.int64); rec = sm["rec"]; fold = sm["fold"]
Q, t, o, has, sbj = g["Q"], g["base"], g["ours"], g["has"], g["sbj"]; S = has
print("hand gate tau grid, F1(S):")
best = {}
for rule in ("top2", "plain"):
    line = []
    for tau in np.round(np.arange(0.40, 0.951, 0.05), 2):
        lab, _ = G.apply_gate(t, Q, o, has, str(tau), rule); f = macro_f1(y[S], lab[S]); line.append(f"{tau:.2f}:{f:.4f}")
        if f > best.get(rule, (0, 0))[0]:
            best[rule] = (f, tau)
    print(f"  {rule}: " + " ".join(line))
print("best hand gates:", best)
r = np.load(os.path.join(OUT, sys.argv[1]))
ref_lab = r["lab"]; gl = g["gate_lab"]
sess = np.unique(rec[S]); rng = np.random.default_rng(0); diffs = []; d_base = []
idx_by = {s: np.flatnonzero(S & (rec == s)) for s in sess}
for b in range(1000):
    pick = rng.choice(sess, len(sess), replace=True); ii = np.concatenate([idx_by[s] for s in pick])
    diffs.append(macro_f1(y[ii], ref_lab[ii]) - macro_f1(y[ii], gl[ii])); d_base.append(macro_f1(y[ii], gl[ii]) - macro_f1(y[ii], t[ii]))
diffs = np.array(diffs); d_base = np.array(d_base)
print(f"session bootstrap (18 sessions, 1000x): refiner - gate mean {diffs.mean():.4f}, 90% CI [{np.quantile(diffs, 0.05):.4f}, {np.quantile(diffs, 0.95):.4f}], P(>0) {np.mean(diffs > 0):.3f}")
print(f"                                     gate - base    mean {d_base.mean():.4f}, 90% CI [{np.quantile(d_base, 0.05):.4f}, {np.quantile(d_base, 0.95):.4f}], P(>0) {np.mean(d_base > 0):.3f}")
per = []
for s in sess:
    m = S & (rec == s); per.append((int(s), int(sbj[m][0]), int(fold[m][0]), macro_f1(y[m], t[m]), macro_f1(y[m], gl[m]), macro_f1(y[m], ref_lab[m])))
print("per session (rec, sbj, fold, base, gate, refiner):")
for p_ in per:
    print("  " + " ".join(str(v) if isinstance(v, int) else f"{v:.4f}" for v in p_) + f"   d={p_[5] - p_[4]:+.4f}")
print(f"sessions where refiner > gate: {sum(p_[5] > p_[4] for p_ in per)}/{len(per)}")
names = [str(x) for x in cv["names"]]; D = S & (t != o); Dt = te["t"] != te["o"]
for nm in ("lrI", "lrF", "rkI_o", "rkF_o", "C_lrW", "L_lrW", "K_lrW", "lrW", "lrP0"):
    j = names.index(nm); a, b = cv["X"][D, j], te["X"][Dt, j]; a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    auc = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
    print(f"shift {nm:6s}: oof mean {a.mean():7.3f} test mean {b.mean():7.3f}  advAUC {max(auc, 1 - auc):.3f}")
