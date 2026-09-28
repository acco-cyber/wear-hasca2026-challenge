"""Compare public submissions against our LB-scored files; extract notebook code cells to .py."""
import os, json, sys
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"
ours = {"e44": "subs/sub_e44_vote9.csv", "e39": "subs/sub_uec3_u3_uec_primary.csv", "e30": "subs/sub_e30_vote8.csv",
        "e19": "subs/sub_transductive_mrf4_e19_aka045_abh.csv"}
pubs = {"woominyo": "public_src/woominyo/out/submission.csv", "aka_chain": "public_src/aka_chain/out/submission.csv",
        "aka_chain_imu": "public_src/aka_chain/out/submission_imu.csv", "aka_chain_video": "public_src/aka_chain/out/submission_video.csv",
        "aka_chain_pose": "public_src/aka_chain/out/submission_pose.csv"}
def load(p):
    d = pd.read_csv(os.path.join(W, p))
    d = d.sort_values(d.columns[0]).reset_index(drop=True)
    return d
from sklearn.metrics import f1_score
ref = {k: load(v) for k, v in ours.items()}
print("columns", ref["e44"].columns.tolist(), len(ref["e44"]))
lab_col = ref["e44"].columns[1]
for k, v in pubs.items():
    d = load(v)
    print(f"\n== {k}: cols {d.columns.tolist()} n={len(d)} null_frac={(d[lab_col]==0).mean() if d[lab_col].dtype!=object else (d[lab_col]=='null').mean():.3f}")
    for rk, rd in ref.items():
        a = (d[lab_col].values == rd[lab_col].values).mean()
        try:
            f = f1_score(rd[lab_col].values, d[lab_col].values, average="macro")
        except Exception as e:
            f = float("nan")
        print(f"   vs {rk}: agree={a:.4f} macroF1={f:.4f}")
    print("   label counts:", d[lab_col].value_counts().head(20).to_dict())
# extract code cells
for nb, out in [("public_src/woominyo/lb-0-890-wear-hasca-2026-timeline-graph.ipynb", "public_src/woominyo/nb_code.py"),
                ("public_src/aka_chain/wear-hasca-chain-lb-0-75.ipynb", "public_src/aka_chain/nb_code.py")]:
    j = json.load(open(os.path.join(W, nb), encoding="utf-8"))
    cells = []
    for c in j["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code":
            cells.append(f"# %% code\n{src}\n")
        else:
            cells.append("# %% markdown\n" + "\n".join("# " + l for l in src.splitlines()) + "\n")
    open(os.path.join(W, out), "w", encoding="utf-8").write("\n".join(cells))
    print("wrote", out, len(j["cells"]), "cells")
