"""Evaluate family disambiguation variants on cached mrf4 outputs."""
import pickle, sys, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from common import mf1, EVAL, NC
from famdis import *
from vidknn import mp_desc, train_vid

C = pickle.load(open("mrf4_cache.pkl", "rb")); SESS = list(C)
rows = []
for s in SESS:
    c = C[s]; y = c["y"]; lab = c["lab"]; L = c["logP"]; W = c["W"]; D = mp_desc(train_vid()[c["a"]:c["b_"]])
    Lb = L + c["b"][None]
    r = dict(session=s, mrf4=mf1(y, lab))
    # oracle grouping (true labels) -> upper bound of the bout-level assignment
    r["oracle"] = mf1(y, family_pass(lab, L, groups_oracle(y)))
    r["current"] = mf1(y, family_pass(lab, L, groups_current))
    for ex in (0, 1, 2):
        r[f"spec+{ex}"] = mf1(y, family_pass(lab, L, groups_spectral(W, ex)))
    r["spec+1_vid"] = mf1(y, family_pass(lab, L, groups_spectral(W, 1, D, 10, 0.1)))
    r["spec+0_bias"] = mf1(y, family_pass(lab, Lb, groups_spectral(W, 0)))
    for res in (0.5, 1.0, 2.0):
        r[f"louv{res}"] = mf1(y, family_pass(lab, L, groups_louvain(W, res)))
    for big in (130, 160):
        r[f"split{big}"] = mf1(y, family_pass(lab, L, groups_split(W, big)))
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows).set_index("session").T
df["EVAL"] = df[EVAL].mean(1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(1)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(df.round(4).to_string()); df.to_csv("exp_fam.csv")
