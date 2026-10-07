"""Gate A: links file for v4_local --links : OOF matchings from a simulated-chain links npz (research2/w25feas/cache),
test matchings copied from the fit's own keep4/links.npz (placeholder, only the OOF decode matters here).
  python a2_mklinks.py <sim_links.npz> <keep4 dir> <out.npz>"""
import os, sys, numpy as np
sim, d, out = sys.argv[1:4]
z = np.load(sim); l = np.load(os.path.join(d, "links.npz")); st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True)
So, Co = z["oof_succ"].astype(np.int64), z["oof_score"].astype(np.float32)
N = len(st["oof_sbj"]); assert So.shape == Co.shape and So.shape[1] == N, (So.shape, N)
sbj = st["oof_sbj"].astype(np.int64)
for k in range(len(So)):                                   # 1:1, within subject, no self links
    s = So[k]; m = s >= 0
    assert len(np.unique(s[m])) == m.sum() and (sbj[s[m]] == sbj[m]).all() and (s[m] != np.flatnonzero(m)).all()
ts = st["true_succ"].astype(np.int64); h = ts >= 0
print("members", len(So), "exact successor per member", np.round([(So[k][h] == ts[h]).mean() for k in range(len(So))], 4),
      "| own member0", round(float((l["oof_succ"][0][h] == ts[h]).mean()), 4))
np.savez(out, oof_succ=So, oof_score=Co, test_succ=l["test_succ"].astype(np.int64), test_score=l["test_score"].astype(np.float32))
print("wrote", out)
