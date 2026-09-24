"""Total activity windows: truth vs e7 decode (protocol: 18 activities x ~95 s)."""
import pickle
import numpy as np
from sklearn.metrics import f1_score
C = pickle.load(open(r"E:\Claude code\wear\exp\order\cache_e7.pkl", "rb"))
for s, d in C.items():
    y = d["y"]; lab = d["lab1"]; n = len(y)
    print(f"{s:9s} n {n} true act {np.sum(y>0)} e7 act {np.sum(lab>0)} (diff {np.sum(lab>0)-np.sum(y>0):+d}) null->act {np.sum((y==0)&(lab>0))} act->null {np.sum((y>0)&(lab==0))} f1 {d['f1']:.4f} "
          f"| min/med/max decoded act count {np.bincount(lab,minlength=19)[1:].min()}/{int(np.median(np.bincount(lab,minlength=19)[1:]))}/{np.bincount(lab,minlength=19)[1:].max()}")
import pandas as pd
st = pickle.load(open(r"E:\Claude code\wear\work\test_structure.pkl", "rb"))
print({s: len(v["idx"]) for s, v in st.items()})
