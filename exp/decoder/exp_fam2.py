"""Oracle-grouping diagnostics per family and per utility source."""
import pickle, numpy as np, pandas as pd
from common import mf1, EVAL
from famdis import *
C = pickle.load(open("mrf4_cache.pkl", "rb")); SESS = list(C); rows = []
for s in SESS:
    c = C[s]; y = c["y"]; lab = c["lab"]; L = c["logP"]; Lb = L + c["b"][None]; Lr = np.log(np.clip(c["P"], 1e-6, 1))
    r = dict(session=s, mrf4=mf1(y, lab))
    for nm, U in (("L", L), ("Lb", Lb), ("raw", Lr)):
        r[f"or_{nm}"] = mf1(y, family_pass(lab, U, groups_oracle(y)))
    for i, F in enumerate(FAMILIES):
        r[f"or_L_fam{F[0]}"] = mf1(y, family_pass(lab, L, groups_oracle(y), fams=[F]))
    # oracle with only true-activity groups of the family (non-family windows keep current label)
    def g_pure(idx, F, lab_):
        g = y[idx].copy(); return g
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows).set_index("session").T
df["EVAL"] = df[EVAL].mean(axis=1); df["EXTRA"] = df[[s for s in SESS if s not in EVAL]].mean(axis=1)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(df.round(4).to_string())
