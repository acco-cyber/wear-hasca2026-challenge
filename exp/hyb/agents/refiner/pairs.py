"""Where does the refiner switch? (their->ours class pairs) OOF vs test, and per test subject.  python pairs.py <oof.npz> <test.csv>"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import KEEP
OUT = os.path.dirname(os.path.abspath(__file__)); W = r"E:\Claude code\wear"
g = np.load(os.path.join(OUT, "graph_cv.npz")); gt = np.load(os.path.join(OUT, "graph_test.npz"))
y = np.load(os.path.join(KEEP, "sim_meta.npz"))["y"]; t, o, S = g["base"], g["ours"], g["has"]
lab = np.load(os.path.join(OUT, sys.argv[1]))["lab"]; sw = S & (lab != t)
labt = pd.read_csv(os.path.join(OUT, sys.argv[2])).sort_values("id").target_feature.to_numpy(); tt, ot, sb = gt["base"], gt["ours"], gt["sbj"]
swt = labt != tt; ref = pd.read_csv(os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv")).sort_values("id").target_feature.to_numpy(); gsw = ref != tt
def kind(a, b):
    return "act->null" if b == 0 else ("null->act" if a == 0 else "act->act")
for nm, m, A, B in (("OOF refiner", sw, t, o), ("TEST refiner", swt, tt, ot), ("TEST hand gate", gsw, tt, ot)):
    c = Counter(kind(a, b) for a, b in zip(A[m], B[m])); tot = m.sum()
    print(f"{nm}: {tot} switches; " + ", ".join(f"{k} {v / tot:.2f}" for k, v in sorted(c.items())))
    print("   top pairs: " + ", ".join(f"{a}->{b}:{v}" for (a, b), v in Counter(zip(A[m].tolist(), B[m].tolist())).most_common(10)))
print("OOF switch accuracy by kind: " + ", ".join(
    f"{k}: ours right {np.mean([y[i] == o[i] for i in np.flatnonzero(sw) if kind(t[i], o[i]) == k]):.3f} theirs right {np.mean([y[i] == t[i] for i in np.flatnonzero(sw) if kind(t[i], o[i]) == k]):.3f}"
    for k in ("act->null", "null->act", "act->act")))
D = S & (t != o); Dt = tt != ot
print("disagree kinds OOF: " + str({k: round(v / D.sum(), 3) for k, v in Counter(kind(a, b) for a, b in zip(t[D], o[D])).items()}))
print("disagree kinds TEST: " + str({k: round(v / Dt.sum(), 3) for k, v in Counter(kind(a, b) for a, b in zip(tt[Dt], ot[Dt])).items()}))
for s in np.unique(sb):
    m = sb == s
    print(f"test sbj {s}: n {m.sum()} disagree {Dt[m].mean():.4f} refiner switched {swt[m].mean():.4f} gate switched {gsw[m].mean():.4f} both {np.mean(swt[m] & gsw[m] & (labt[m] == ref[m])):.4f}")
