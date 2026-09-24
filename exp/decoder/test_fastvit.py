import time, numpy as np
from common import *
from fastvit import *
oof = load_oof(0.2); S = load_structs(("eval",))
for s in EVAL[:3]:
    st = S[s]; P = sess_P(oof, st); n = st["n"]
    Pg = smooth(P, graph_matrix(st["cand"], st["lo"], n, k=10), 0.5, 5); ch = base_chains(st)
    t0 = time.time(); l1, b1 = calibrate_counts(Pg, ch, lo=60, hi=160, p_stay=0.8); t1 = time.time()
    cb = ChainBatch(ch, n); l2, b2 = calib_decode(np.log(np.clip(Pg, 1e-6, 1)), cb, 60, 160, 0.8); t2 = time.time()
    print(s, "agree", (l1 == l2).mean(), "f1", mf1(st["y"], l1), mf1(st["y"], l2), "time", t1 - t0, t2 - t1, "chains", len(ch), "maxlen", cb.L)
