"""Error anatomy of the fused b4wa decode: confidence, top-k coverage, oracle of the bottom-10% re-judge."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1

W = r"E:\Claude code\wear"
K7 = W + r"\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = W + r"\work\v4\wear-v4-big-tf-opt-s9\keep4"
st = np.load(K7 + r"\stage.npz", allow_pickle=True)
y = st["oof_y"].astype(int); sbj = st["oof_sbj"].astype(int); fold = st["oof_fold"].astype(int); tsbj = st["test_sbj"].astype(int)
Qo = np.load(W + r"\subs\sub_v4c_b4wa_Qo.npy").astype(np.float64); Qt = np.load(W + r"\subs\sub_v4c_b4wa_Qt.npy").astype(np.float64)
lo = np.load(W + r"\subs\sub_v4c_b4wa_labo.npy").astype(int); lt = np.load(W + r"\subs\sub_v4c_b4wa_labt.npy").astype(int)
fo = Qo.argmax(1)
print("Qo argmax", round(macro_f1(y, fo), 4), "refined", round(macro_f1(y, lo), 4), "acc", (lo == y).mean().round(4))
print("refiner changed", (lo != fo).sum(), "OOF tiles;", (lt != Qt.argmax(1)).sum(), "test tiles")
conf = Qo.max(1); ct = Qt.max(1)
print("conf quantiles oof", np.quantile(conf, [.05, .1, .2, .3, .5]).round(3), "test", np.quantile(ct, [.05, .1, .2, .3, .5]).round(3))
err = lo != y
print("errors", err.sum(), "rate", err.mean().round(4))
rk = np.argsort(-Qo, 1)
for k in (1, 2, 3, 4):
    cov = (rk[:, :k] == y[:, None]).any(1)
    print(f"top{k} of Qo covers truth: all {cov.mean():.4f}, among refined errors {cov[err].mean():.4f}")
# per-subject bottom 10% confidence
for q in (0.05, 0.1, 0.2, 0.3):
    low = np.zeros(len(y), bool)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); low[ii[conf[ii] <= np.quantile(conf[ii], q)]] = True
    orc = lo.copy(); orc[low] = y[low]
    orc2 = lo.copy(); m2 = low & (rk[:, :3] == y[:, None]).any(1); orc2[m2] = y[m2]
    print(f"bottom {q:.0%} per subject: {low.sum()} tiles, {err[low].sum()} of {err.sum()} errors ({err[low].mean():.3f} err rate);"
          f" oracle fix -> {macro_f1(y, orc):.4f}; oracle within top3 -> {macro_f1(y, orc2):.4f}")
# null vs activity error anatomy
cm = np.bincount(y[err] * 19 + lo[err], minlength=361).reshape(19, 19)
print("errors: truth null->act", cm[0, 1:].sum(), "act->null", cm[1:, 0].sum(), "act->act", cm[1:, 1:].sum())
# targets recovered from Qo column sums vs true counts
for s in np.unique(sbj)[:3]:
    ii = sbj == s
    print(s, Qo[ii].sum(0).round(1)[:6], np.bincount(y[ii], minlength=19)[:6])
print("per-fold refined", [round(macro_f1(y[fold == f], lo[fold == f]), 4) for f in range(5)])
e7 = np.load(K7 + r"\oof_emb.npy", mmap_mode="r"); e9 = np.load(K9 + r"\oof_emb.npy", mmap_mode="r")
print("emb", e7.shape, "same in K7/K9:", np.allclose(e7[:2000], e9[:2000]))
lk = np.load(K7 + r"\links.npz"); su = lk["oof_succ"]; ts = st["true_succ"]
m = su[0] >= 0
print("matching0 exact successor", (su[0][m] == ts[m]).mean().round(4), "linked frac", m.mean().round(4), "true has succ", (ts >= 0).mean().round(4))
print("same-label successor (m0)", (y[su[0][m]] == y[m]).mean().round(4))
