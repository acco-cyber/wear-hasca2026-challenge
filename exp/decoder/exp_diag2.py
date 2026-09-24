"""Error structure after the MRF decoder (lam=3)."""
import numpy as np, scipy.sparse as sp
from sklearn.metrics import confusion_matrix
from common import *
from nbvit import pack
from mrf import calib_mrf
from harness import data
oof, S = data(); SESS = [s for s in EVAL + EXTRA if s in S]
fam = np.array([0] + [1] * 5 + [2] * 5 + [3, 3, 4, 4, 5, 6, 6, 7])
CM = np.zeros((NC, NC), int)
for s in SESS:
    st = S[s]; P = sess_P(oof, st); y = st["y"]; n = st["n"]
    W0 = graph_matrix(st["cand"], st["lo"], n, k=10); logP = np.log(np.clip(smooth(P, W0, 0.5, 5), 1e-6, 1))
    W = W0 + W0.T; d = np.asarray(W.sum(1)).ravel(); W = (sp.diags(1 / np.where(d > 0, d, 1)) @ W).tocsr()
    lab, b = calib_mrf(logP, pack(base_chains(st)), W, 85, 160, 0.8, 3.0, 10)
    cm = confusion_matrix(y, lab, labels=np.arange(NC)); CM += cm; off = cm.copy(); np.fill_diagonal(off, 0)
    within = sum(off[i, j] for i in range(1, NC) for j in range(1, NC) if fam[i] == fam[j])
    print(s, "f1 %.4f" % mf1(y, lab), "err", off.sum(), "null->act", off[0, 1:].sum(), "act->null", off[1:, 0].sum(), "act->act", off[1:, 1:].sum(), "(within fam", within, ")",
          "| pred null %.3f true %.3f" % ((lab == 0).mean(), (y == 0).mean()), "| bias0 %.2f" % b[0], flush=True)
    # per-class: count pred vs true
    print("   pred", np.bincount(lab, minlength=NC)[1:].tolist()); print("   true", np.bincount(y, minlength=NC)[1:].tolist())
np.set_printoptions(linewidth=250); print(CM)
