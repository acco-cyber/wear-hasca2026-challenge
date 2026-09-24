import pickle, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\order")
from olib import SESS, ALL
r = pickle.load(open(sys.argv[1], "rb"))["R"]
keys = list(r[ALL[0]])
print("session   " + " ".join(f"k{i:<6d}" for i in range(len(keys))))
for s in ALL: print(f"{s:9s} " + " ".join(f"{r[s][k]:.4f} " for k in keys))
for i, k in enumerate(keys): print(f"k{i}: {k}")
