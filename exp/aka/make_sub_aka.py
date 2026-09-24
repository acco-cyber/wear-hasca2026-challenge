"""Writes subs/sub_aka_raw.csv = argmax of the full-fit akhyar reproduction (reference only; the lead blends test.npy)."""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
T = np.load(os.path.join(HERE, "aka_full", "test.npy")); assert T.shape == (12234, 19)
out = r"E:\Claude code\wear\subs\sub_aka_raw.csv"
pd.DataFrame({"id": np.arange(len(T)), "target_feature": T.argmax(1)}).to_csv(out, index=False)
print("wrote", out)
