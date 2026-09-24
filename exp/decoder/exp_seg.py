"""Segment ILP decoder experiments."""
import sys, time, numpy as np
from common import *
from segdec import *
from harness import data, run

oof, S = data()
_c = {}
def prep(P, st):
    k = id(st)
    if k not in _c:
        n = st["n"]; R, C, Lo = graph_edges(st["cand"], st["lo"], k=10)
        W = graph_matrix(st["cand"], st["lo"], n, k=10); Pg = smooth(P, W, 0.5, 5); ch = base_chains(st)
        lab0 = calibrate_counts(Pg, ch, lo=60, hi=160, p_stay=0.8)[0]
        _c[k] = dict(R=R, C=C, Lo=Lo, Pg=Pg, ch=ch, lab0=lab0)
    return _c[k]

# segmentation quality
for s in EVAL[:3]:
    st = S[s]; P = sess_P(oof, st); d = prep(P, st)
    for tau in (-6, -4, -2, 0):
        seg = segments_from_labels(d["lab0"], d["R"], d["C"], d["Lo"], st["n"], tau)
        print(s, tau, seg_stats(seg, st["y"]), flush=True)

def mk(tau, lo, hi, mu):
    def f(P, st):
        d = prep(P, st); seg = segments_from_labels(d["lab0"], d["R"], d["C"], d["Lo"], st["n"], tau)
        return seg_ilp(np.log(np.clip(d["Pg"], 1e-6, 1)), seg, lo=lo, hi=hi, mu=mu)
    return f
dec = {"base": lambda P, st: prep(P, st)["lab0"]}
for tau in (-4, 0):
    for lo, hi in ((75, 130), (85, 120)):
        dec[f"t{tau}_{lo}_{hi}_mu2"] = mk(tau, lo, hi, 2.0)
dec["t-4_85_120_mu5"] = mk(-4, 85, 120, 5.0)
dec["win_85_120_mu2"] = lambda P, st: seg_ilp(np.log(np.clip(prep(P, st)["Pg"], 1e-6, 1)), np.arange(st["n"]), 85, 120, 2.0)
run(dec, tag="segment ILP")
