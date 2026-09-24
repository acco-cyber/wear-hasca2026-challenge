import pickle
import numpy as np, pandas as pd
tm = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
for f in ["sub_transductive_mrf4_knn_w6_d128.csv", "sub_e10_e7_plus_block4.csv"]:
    sub = pd.read_csv(rf"E:\Claude code\wear\subs\{f}")
    print(f)
    for s, g in tm.groupby("sbj_id"):
        lab = sub.target_feature.to_numpy()[g.id.to_numpy()]; c = np.bincount(lab, minlength=19)
        print(f"  sbj {s} n {len(g)} act {c[1:].sum()} null {c[0]} counts {c[1:].tolist()}")
