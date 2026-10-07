"""proxy check of the TEST chains: for the fit's own L3 links A->B between our tiles with the same limb, how often is B's
twin the chain successor of A's twin; the same statistic on OOF (nested sim chains, K7 OOF links) as the reference."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); lk = np.load(os.path.join(K7, "links.npz"))
# OOF
sens = st["sensor_oof"].astype(np.int64); sbj = st["oof_sbj"].astype(np.int64); ts = st["true_succ"].astype(np.int64)
su0 = lk["oof_succ"][0].astype(np.int64); sc0 = lk["oof_score"][0]
hit = np.zeros(len(sens), np.int8); cnt = np.zeros(len(sens), bool); cs = np.zeros(len(sens), np.float32)
for s in np.unique(sbj):
    z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); rows = z["rows"]; pos = np.full(len(sens), -1); pos[rows] = np.arange(len(rows))
    ii = rows[(su0[rows] >= 0)]; jj = su0[ii]; same = sens[ii] == sens[jj]; ii, jj = ii[same], jj[same]
    for L in range(4):
        m = sens[ii] == L; a, b = ii[m], jj[m]
        hit[a] = (z[f"succ{L}"][pos[a]] == pos[b]); cnt[a] = True; cs[a] = z[f"conf{L}"][pos[a]]
q = np.quantile(sc0[cnt], [0.5])[0]
print(f"OOF: same-limb own links {cnt.sum()}; chain agrees {hit[cnt].mean():.4f} (links with score>median {hit[cnt & (sc0 > q)].mean():.4f}); "
      f"link exact {np.mean(su0[cnt] == ts[cnt]):.4f}; per limb " + " ".join(f"L{L} {hit[cnt & (sens == L)].mean():.4f}" for L in range(4)))
# TEST
tz = np.load(os.path.join(CACHE, "test_twins.npz")); twin = tz["twin"]
sens_t = st["sensor_test"].astype(np.int64); tsb = st["test_sbj"].astype(np.int64)
su = lk["test_succ"][0].astype(np.int64); sc = lk["test_score"][0]
hit = np.zeros(len(sens_t), np.int8); cnt = np.zeros(len(sens_t), bool)
for s in (22, 23, 24, 25):
    z = np.load(os.path.join(CACHE, f"test_chain_s{s}.npz"))
    for L in range(4):
        rows = z[f"rows{L}"]; pos = np.full(48936, -1); pos[rows] = np.arange(len(rows))
        a = np.flatnonzero((tsb == s) & (sens_t == L) & (su >= 0)); b = su[a]; m = sens_t[b] == L; a, b = a[m], b[m]
        hit[a] = (z[f"succ{L}"][pos[twin[a]]] == pos[twin[b]]); cnt[a] = True
qt = np.quantile(sc[cnt], [0.5])[0]
print(f"TEST: same-limb own links {cnt.sum()}; chain agrees {hit[cnt].mean():.4f} (links with score>median {hit[cnt & (sc > qt)].mean():.4f}); per limb "
      + " ".join(f"L{L} {hit[cnt & (sens_t == L)].mean():.4f}" for L in range(4)))
