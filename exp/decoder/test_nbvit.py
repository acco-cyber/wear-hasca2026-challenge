import time, numpy as np
from common import *
from nbvit import *
oof = load_oof(0.2); S = load_structs(("eval",))
for s in EVAL:
    st = S[s]; P = sess_P(oof, st); n = st["n"]
    Pg = smooth(P, graph_matrix(st["cand"], st["lo"], n, k=10), 0.5, 5); ch = base_chains(st)
    t0 = time.time(); l1, b1 = calibrate_counts(Pg, ch, lo=60, hi=160, p_stay=0.8); t1 = time.time()
    l2, b2 = calib(np.log(np.clip(Pg, 1e-6, 1)), pack(ch), 60, 160, 0.8); t2 = time.time()
    print(s, "agree", (l1 == l2).mean(), "f1", round(mf1(st["y"], l1), 4), round(mf1(st["y"], l2), 4), "time", round(t1 - t0, 2), round(t2 - t1, 2))
