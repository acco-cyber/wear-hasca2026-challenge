"""Within-family count balance: true per-class counts per session and e7 decoded counts."""
import pickle
import numpy as np, pandas as pd
meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
FAMS = {"jog": [1, 2, 3, 4, 5], "str": [6, 7, 8, 9, 10], "push": [11, 12], "sit": [13, 14], "lun": [16, 17]}
rows = []
for s, g in meta.groupby("session", sort=False):
    c = np.bincount(g.y.to_numpy(), minlength=19)
    r = dict(session=s, n=len(g), act=int(c[1:].sum()))
    for f, cls in FAMS.items():
        v = c[cls]; r[f] = "/".join(str(int(x)) for x in v); r[f + "_ratio"] = round(v.max() / max(v.min(), 1), 2)
    r["all_cv"] = round(np.std(c[1:]) / np.mean(c[1:]), 2)
    if s in R:
        e = np.bincount(R[s]["lab"], minlength=19)
        for f, cls in FAMS.items(): r[f + "_e7"] = "/".join(str(int(x)) for x in e[cls])
    rows.append(r)
df = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(df[["session", "n", "act", "all_cv", "push", "push_e7", "sit", "sit_e7", "lun", "lun_e7"]].to_string())
print(df[["session", "jog", "jog_e7", "str", "str_e7"]].to_string())
print(df[[c for c in df.columns if c.endswith("_ratio")]].describe().round(2))
