"""Is the link headroom reachable by REMOVING harmful links (a label-free detector) rather than finding the true
successor? K7, OOF only, learned nested counts.
  oracle-drop : every member's links to a tile with another TRUE label are unlinked (no replacement)  [ORACLE]
  learned-dropT: a LightGBM detector (trained on the other 4 subject folds) predicts 'cross-label link' from label-free
                 features of the two tiles; links with p > T are unlinked (T = 0.5 a priori; 0.3 reported as sensitivity)"""
import sys
import numpy as np
from hlib import *

D = setup(); y, fold = D["y"], D["fold"]; F = D["F"]
z = np.load(os.path.join(HERE, "k7_base.npz")); Qb = z["Q"].astype(np.float64); Pb = z["P"].astype(np.float64)
ener, post, vmot, vmean = F["sc_o"]; sens = F["sensor_oof"].astype(np.int64)
B = np.exp(D["Bo"].astype(np.float64)); lB = D["Bo"].astype(np.float64)
vm = vmean / (np.linalg.norm(vmean, axis=1, keepdims=True) + 1e-6)


def feats(i, j, sc):
    """label-free features of link i -> j (base decode Q/P of the ORIGINAL K7 run, stage-B base, tile scalars)"""
    fi, fj = Qb[i].argmax(1), Qb[j].argmax(1)
    return np.c_[sc, (fi == fj), (Qb[i] * Qb[j]).sum(1), np.abs(Qb[i] - Qb[j]).sum(1), (Pb[i] * Pb[j]).sum(1),
                 np.abs(B[i] - B[j]).sum(1), (B[i] * B[j]).sum(1), Qb[i, 0], Qb[j, 0],
                 B[i, 0], B[j, 0], (fi == 0), (fj == 0), np.abs(ener[i] - ener[j]), vmot[i], vmot[j],
                 (vm[i] * vm[j]).sum(1), (sens[i] == sens[j]), np.where(sens[i] == sens[j], np.linalg.norm(post[i] - post[j], axis=1), -1.0),
                 Qb[i].max(1), Qb[j].max(1)]


mode = sys.argv[1]
res = {}
if mode == "oracle":
    L = []
    for su, sc in D["Lo"]:
        su, sc = su.copy(), sc.copy(); m = (su >= 0) & (y[np.maximum(su, 0)] != y); su[m] = -1; sc[m] = -50.0; L.append((su, sc))
    log(f"oracle drop: member0 unlinked {np.mean(L[0][0] < 0):.3f}")
    r = decode(D, L, "learned"); ref = refine_oof(D, r["fin"], L, r["P"], r["Q"])
    res["oracle-drop"] = (macro_f1(y, r["fin"]), macro_f1(y, ref))
    np.savez_compressed(os.path.join(HERE, "k7_links_droporacle.npz"), fin=r["fin"], ref=ref)
else:
    import lightgbm as lgb
    thrs = [float(t) for t in sys.argv[2].split(",")]
    rows_i, rows_j, rows_s, rows_k = [], [], [], []
    for k, (su, sc) in enumerate(D["Lo"]):
        ii = np.flatnonzero(su >= 0); rows_i.append(ii); rows_j.append(su[ii]); rows_s.append(sc[ii]); rows_k.append(np.full(len(ii), k))
    I, J, S, K = (np.concatenate(a) for a in (rows_i, rows_j, rows_s, rows_k))
    X = feats(I, J, S).astype(np.float32); T = (y[I] != y[J]).astype(int); Fo = fold[I]; pr = np.zeros(len(T))
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
    for f_ in range(FOLDS):
        tr = Fo != f_
        pr[~tr] = lgb.train(prm, lgb.Dataset(X[tr], T[tr]), 300).predict(X[~tr])
    from sklearn.metrics import roc_auc_score
    log(f"detector: {len(T)} links, cross-label rate {T.mean():.4f}, nested AUC {roc_auc_score(T, pr):.4f}")
    for t in thrs:
        sel = pr > t
        log(f"  p>{t}: drop {sel.mean():.4f} of links; precision {T[sel].mean() if sel.any() else 0:.3f}, recall {sel[T == 1].mean():.3f}")
    for t in thrs:
        L = []
        for k, (su, sc) in enumerate(D["Lo"]):
            su, sc = su.copy(), sc.copy(); m = (K == k) & (pr > t); su[I[m]] = -1; sc[I[m]] = -50.0; L.append((su, sc))
        r = decode(D, L, "learned"); ref = refine_oof(D, r["fin"], L, r["P"], r["Q"])
        res[f"learned-drop{t}"] = (macro_f1(y, r["fin"]), macro_f1(y, ref))
        np.savez_compressed(os.path.join(HERE, f"k7_links_drop{t}.npz"), fin=r["fin"], ref=ref)
for k_, v in res.items():
    log(f"RESULT {k_:20s} pre-refiner {v[0]:.4f} refined {v[1]:.4f}")
