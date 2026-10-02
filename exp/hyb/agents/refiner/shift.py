"""OOF vs test shift on the disagree rows: per-feature standardised mean gap + adversarial AUC per feature and overall."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, lightgbm as lgb
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1
from sklearn.metrics import roc_auc_score
OUT = os.path.dirname(os.path.abspath(__file__))
cv = np.load(os.path.join(OUT, "feats_cv.npz")); te = np.load(os.path.join(OUT, "feats_test.npz"))
names = [str(x) for x in cv["names"]]
S = cv["has"]; y = cv["y"]
print(f"OOF F1(S): theirs {macro_f1(y[S], cv['t'][S]):.4f}, ours {macro_f1(y[S], cv['o'][S]):.4f}; acc theirs {np.mean(y[S] == cv['t'][S]):.4f} ours {np.mean(y[S] == cv['o'][S]):.4f}")
D = S & (cv["t"] != cv["o"]); Dt = te["has"] & (te["t"] != te["o"])
A, B = cv["X"][D], te["X"][Dt]
rows = []
for j, nm in enumerate(names):
    a, b = A[:, j], B[:, j]; a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        continue
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]; auc = roc_auc_score(lab, np.r_[a, b]); auc = max(auc, 1 - auc)
    rows.append((auc, nm, a.mean(), b.mean(), np.mean(np.isnan(A[:, j])), np.mean(np.isnan(B[:, j]))))
for auc, nm, ma, mb, na, nb in sorted(rows, reverse=True):
    print(f"{nm:12s} advAUC {auc:.3f}  mean oof {ma:8.3f} test {mb:8.3f}  nan oof {na:.2f} test {nb:.2f}")
X = np.r_[A, B]; lab = np.r_[np.zeros(len(A)), np.ones(len(B))]
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); fo = idx % 5; p = np.zeros(len(X))
for f in range(5):
    bst = lgb.train(dict(objective="binary", num_leaves=15, learning_rate=0.05, verbose=-1, num_threads=4, min_data_in_leaf=30), lgb.Dataset(X[idx[fo != f]], lab[idx[fo != f]]), 200)
    p[idx[fo == f]] = bst.predict(X[idx[fo == f]])
print(f"overall adversarial AUC (all features) {roc_auc_score(lab, p):.3f}")
