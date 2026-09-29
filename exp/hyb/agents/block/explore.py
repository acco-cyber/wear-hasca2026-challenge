import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import finish, KEEP, TRAIN_SETS, macro_f1, N_CLS
sm = {k: v for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
for k, v in sm.items():
    print(k, v.shape, v.dtype, v[:5], len(np.unique(v)))
sm = {k: v.astype(np.int64) for k, v in sm.items()}
y, rec, st, sbj, fold, sen = sm["y"], sm["rec"], sm["start"], sm["sbj"], sm["fold"], sm["sensor"]
P = np.load(r"E:\Claude code\wear\work\hanbat\cv_all5_oof_P.npy").astype(np.float64); P /= P.sum(1, keepdims=True)
pred = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
print("baseline", macro_f1(y, pred), [round(macro_f1(y[fold == f], pred[fold == f]), 4) for f in range(5)])
print("rec->sbj", {int(r): int(np.unique(sbj[rec == r])[0]) for r in np.unique(rec)})
print("rec->fold", {int(r): np.unique(fold[rec == r]).tolist() for r in np.unique(rec)})
print("sensor counts", np.bincount(sen))
for r in np.unique(rec)[:3]:
    ii = np.flatnonzero(rec == r); ss = np.sort(st[ii])
    print(r, len(ii), "dup starts", len(ii) - len(np.unique(ss)), "min diff", np.diff(ss)[:10], np.unique(sen[ii]))
fo = np.load(os.path.join(KEEP, "feat_oof.npz")); print({k: fo[k].shape for k in fo.files})
ft = np.load(os.path.join(KEEP, "feat_test.npz")); print({k: ft[k].shape for k in ft.files})
bl = np.load(os.path.join(KEEP, "blend.npz")); print({k: bl[k].shape for k in bl.files})
l0 = np.load(os.path.join(KEEP, "links_L0.npz")); print({k: l0[k].shape for k in l0.files})
l2 = np.load(os.path.join(KEEP, "links_L2_test.npz")); print({k: l2[k].shape for k in l2.files})
print("test sbj", np.unique(bl["test_sbj"], return_counts=True))
