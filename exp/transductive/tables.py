"""Print per-session markdown tables for REPORT.md from the results CSVs."""
import json, pandas as pd, numpy as np
TD = r"E:\Claude code\wear\exp\transductive\\"
EVAL = ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"]; EXTRA = ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"]
EXTRA2 = ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]; DEC = ["sbj_2", "sbj_8", "sbj_13", "sbj_17", "sbj_4", "sbj_11"]

def table(csv, match, rnd=1):
    d = pd.read_csv(TD + csv); rows = d[d.cfg.apply(lambda c: all(json.loads(c).get(k) == v for k, v in match.items()) if "steps" not in match else json.loads(c) == match)]
    b = rows[rows["round"] == 0].iloc[0]; r = rows[rows["round"] == rnd].iloc[0]
    print("| session | set | baseline | refined | delta |\n|---|---|---|---|---|")
    for grp, ss in (("eval", EVAL), ("extra", EXTRA), ("extra2", EXTRA2)):
        for s in ss: print(f"| {s} | {grp} | {b[s]:.4f} | {r[s]:.4f} | {r[s]-b[s]:+.4f} |")
    for grp, ss in (("**eval mean**", EVAL), ("**extra mean**", EXTRA), ("**extra2 mean**", EXTRA2), ("**decoder-agent extra (2,8,13,17,4,11)**", DEC), ("**all 18**", EVAL + EXTRA + EXTRA2)):
        bm = np.mean([b[s] for s in ss]); rm = np.mean([r[s] for s in ss]); print(f"| {grp} | | {bm:.4f} | {rm:.4f} | {rm-bm:+.4f} |")
    print()

print("## lead decoder, lgbm_v1/fusion_v1 0.8/0.2"); table("results_g7.json.csv", dict(clf="knn", w=6.0, d_vid=128, k=5, alpha=0.9))
print("## mrf4, bl_v3b_v1_f"); table("results_mrf4_v3b.csv", {"steps": [{"w": 6.0, "d_vid": 128, "k": 5, "alpha": 0.9}]})
