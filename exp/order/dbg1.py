import pickle, sys, json
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0, r"E:\Claude code\wear\exp\order")
from olib import family_bounds
r = pickle.load(open(r"E:\Claude code\wear\exp\order\res_g1.pkl", "rb")); C = pickle.load(open(r"E:\Claude code\wear\exp\order\cache_e7.pkl", "rb"))
for s, k in (("sbj_4", '{"a": 0.75, "b": 1.33, "mode": "fam"}'), ("sbj_0_2", '{"a": 0.85, "b": 1.2, "mode": "fam"}'), ("sbj_10", '{"a": 0.0, "b": 1.3, "mode": "fam"}')):
    y = C[s]["y"]; l0 = C[s]["lab1"]; l1 = r["labs"][s][k]
    print(s, k, "f1", round(f1_score(y, l0, average="macro"), 4), "->", round(f1_score(y, l1, average="macro"), 4))
    print("  true ", np.bincount(y, minlength=19).tolist()); print("  e7   ", np.bincount(l0, minlength=19).tolist()); print("  new  ", np.bincount(l1, minlength=19).tolist())
    c = json.loads(k); lo, hi = family_bounds(l0, c["a"], c["b"], mode=c["mode"]); print("  lo", lo.astype(int).tolist()); print("  hi", hi.astype(int).tolist())
    f0 = f1_score(y, l0, average=None, labels=range(19)); f1 = f1_score(y, l1, average=None, labels=range(19))
    print("  per-class dF1", np.round(f1 - f0, 2).tolist())
