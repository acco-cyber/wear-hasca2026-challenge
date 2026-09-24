"""Practical groupings with bias-applied utilities (Lb)."""
import pickle, numpy as np, pandas as pd
from common import mf1, EVAL, graph_edges, load_structs
from famdis import *
C = pickle.load(open("mrf4_cache.pkl", "rb")); SESS = list(C); S = load_structs(("eval", "extra")); rows = []
for s in SESS:
    c = C[s]; y = c["y"]; lab = c["lab"]; L = c["logP"]; Lb = L + c["b"][None]; W = c["W"]
    R, Cc, Lo = graph_edges(S[s]["cand"], S[s]["lo"], k=10)
    r = dict(session=s, mrf4=mf1(y, lab), or_Lb=mf1(y, family_pass(lab, Lb, groups_oracle(y))))
    for tau in (-2.0, 0.0, 2.0, 4.0):
        g = groups_cc(R, Cc, Lo, tau)
        r[f"cc{tau}"] = mf1(y, family_pass(lab, Lb, g))
        r[f"cc{tau}_mp1"] = mf1(y, family_pass(lab, Lb, g, max_per=1))
    for big in (100, 130):
        r[f"split{big}_Lb"] = mf1(y, family_pass(lab, Lb, groups_split(W, big)))
    r["louv1_Lb"] = mf1(y, family_pass(lab, Lb, groups_louvain(W, 1.0)))
    for lam in (6.0, 10.0, 16.0):
        r[f"famicm{lam}"] = mf1(y, fam_icm(lab, Lb, W, lam))
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows).set_index("session").T
df["EVAL"] = df[EVAL].mean(axis=1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(axis=1)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(df.round(4).to_string()); df.to_csv("exp_fam3.csv")
