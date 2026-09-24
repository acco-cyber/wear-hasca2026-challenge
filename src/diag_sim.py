"""Diagnostics of the baseline decoder on sim sessions: null rate bias, per-class F1, effect of null_scale."""
import os, sys, pickle, numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from chain import cut, chains_from_succ
from decode import calibrate_counts, build_graph, graph_smooth

W = r"E:\Claude code\wear\work"
S = pickle.load(open(os.path.join(W, "sim_struct.pkl"), "rb"))
A = np.load(os.path.join(W, "lgbm_v1", "oof.npy")); B = np.load(os.path.join(W, "fusion_v1", "oof.npy"))
oof = np.exp(0.8 * np.log(np.clip(A, 1e-6, 1)) + 0.2 * np.log(np.clip(B, 1e-6, 1))); oof /= np.nansum(oof, 2, keepdims=True)
ys, ps = [], []
rows = []
for s, st in S.items():
    n = st["n"]; y = st["y"]
    P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]]; ok = ~np.isnan(P[:, 0]); P = np.where(ok[:, None], P, 1.0 / 19)
    g = build_graph(st["cand"], st["lo"], n, k=10); chains = chains_from_succ(cut(st["succ0"], st["sc"], st["Lm"], -6.0))
    res = dict(session=s, n=n, null_true=round((y == 0).mean(), 3))
    for ns in (1.0, 0.7, 0.5):
        Pn = P.copy(); Pn[:, 0] *= ns; Pn /= Pn.sum(1, keepdims=True)
        Pg = graph_smooth(Pn, g, alpha=0.5, iters=5)
        for lo_, hi_ in ((60, 160), (70, 200)):
            lab, _ = calibrate_counts(Pg, chains, lo=lo_, hi=hi_, p_stay=0.8)
            res[f"f1_ns{ns}_{lo_}_{hi_}"] = round(f1_score(y, lab, average="macro"), 4)
            res[f"null_ns{ns}_{lo_}_{hi_}"] = round((lab == 0).mean(), 3)
            if ns == 1.0 and lo_ == 60: ys.append(y); ps.append(lab)
    act = np.bincount(y, minlength=19)[1:]
    res["true_act_sec_med"] = int(np.median(act)); res["true_act_sec_min"] = int(act.min()); res["true_act_sec_max"] = int(act.max())
    rows.append(res)
pd.set_option("display.width", 300); df = pd.DataFrame(rows); print(df.T.to_string())
print(df.mean(numeric_only=True).round(4).to_string())
y = np.concatenate(ys); p = np.concatenate(ps)
f = f1_score(y, p, average=None, labels=range(19))
print("per-class F1 (pooled 6 sessions, baseline decoder):", np.round(f, 3).tolist())
cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(p, name="pred"))
print(cm.to_string())
