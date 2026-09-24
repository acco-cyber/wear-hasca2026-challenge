import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
SESS = {"eval": ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"],
        "extra": ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"],
        "extra2": ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]}
print(list(R["sbj_0"].keys()))
for w, ss in SESS.items():
    print(w, np.mean([R[s]["f1"] for s in ss]).round(4), [round(R[s]["f1"], 4) for s in ss])
print("all18", np.mean([R[s]["f1"] for s in sum(SESS.values(), [])]).round(4))
meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
print(meta.head()); print(meta.session.unique())
# segments of true labels per session in time order
for s in meta.session.unique():
    y = meta[meta.session == s].y.to_numpy()
    ch = np.r_[0, np.where(np.diff(y) != 0)[0] + 1]
    seq = [(int(y[i]), int((np.r_[ch, len(y)][k + 1]) - i)) for k, i in enumerate(ch)]
    acts = [a for a, l in seq if a > 0]
    # compress consecutive same activity (sets separated by null)
    comp = [acts[0]] + [a for p, a in zip(acts[:-1], acts[1:]) if a != p]
    nulls = [l for a, l in seq if a == 0]
    print(s, len(y), "order:", comp, "| n act segs", len(acts), "| null seg lens med", int(np.median(nulls)) if nulls else 0)
