"""Agreement / macro-F1 between submission files. python agree.py ref.csv other.csv [other.csv ...]"""
import sys, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
def L(p):
    return pd.read_csv(p).sort_values("id").target_feature.values
ref = L(sys.argv[1])
for p in sys.argv[2:]:
    if not os.path.exists(p):
        print(os.path.basename(p), "missing"); continue
    x = L(p)
    print(f"{os.path.basename(p):36s} vs {os.path.basename(sys.argv[1])}: agree {np.mean(x == ref):.4f} F1 {f1_score(ref, x, average='macro'):.4f} null {np.mean(x == 0):.3f}")
