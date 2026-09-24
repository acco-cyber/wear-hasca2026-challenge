"""Where are the null<->activity errors of mrf4? distance to true boundary, purity, decoded-neighbour agreement."""
import pickle, numpy as np, pandas as pd
from common import PREP
import os
meta = pd.read_csv(os.path.join(PREP, "train_meta.csv")); pur_all = meta.pur.to_numpy()
C = pickle.load(open("mrf4_cache.pkl", "rb")); tot = dict(n2a=0, a2n=0, n2a_b2=0, a2n_b2=0, n2a_b5=0, a2n_b5=0, n2a_imp=0, a2n_imp=0)
for s, c in C.items():
    y = c["y"]; lab = c["lab"]; n = len(y); pur = pur_all[c["a"]:c["b_"]]
    ch = np.r_[False, y[1:] != y[:-1]]; bpos = np.where(ch)[0]
    d = np.abs(np.arange(n)[:, None] - bpos[None, :]).min(1) if len(bpos) else np.full(n, 999)
    n2a = (y == 0) & (lab > 0); a2n = (y > 0) & (lab == 0)
    tot["n2a"] += n2a.sum(); tot["a2n"] += a2n.sum()
    tot["n2a_b2"] += (n2a & (d <= 2)).sum(); tot["a2n_b2"] += (a2n & (d <= 2)).sum()
    tot["n2a_b5"] += (n2a & (d <= 5)).sum(); tot["a2n_b5"] += (a2n & (d <= 5)).sum()
    tot["n2a_imp"] += (n2a & (pur < 0.8)).sum(); tot["a2n_imp"] += (a2n & (pur < 0.8)).sum()
    # null runs between two same-label activity runs (rest between sets)
print(tot)
# length of true null gaps inside the same activity (between sets) and how they are decoded
rows = []
for s, c in C.items():
    y = c["y"]; lab = c["lab"]; n = len(y)
    runs = []; st = 0
    for i in range(1, n + 1):
        if i == n or y[i] != y[st]: runs.append((y[st], st, i)); st = i
    for j in range(1, len(runs) - 1):
        if runs[j][0] == 0 and runs[j - 1][0] == runs[j + 1][0] and runs[j - 1][0] > 0:
            a, b = runs[j][1], runs[j][2]; rows.append(dict(s=s, len=b - a, dec_null=(lab[a:b] == 0).mean(), dec_same=(lab[a:b] == runs[j - 1][0]).mean()))
df = pd.DataFrame(rows); print("between-set null gaps:", len(df), "windows", df.len.sum(), "decoded null frac (weighted)", round((df.dec_null * df.len).sum() / df.len.sum(), 3),
                              "decoded as the activity", round((df.dec_same * df.len).sum() / df.len.sum(), 3))
