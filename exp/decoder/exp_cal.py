"""Sweep count-calibration bounds / p_stay of the baseline chain decoder."""
import sys, numpy as np
from common import *
from harness import run

_gcache = {}
def Pg_of(P, st, k=10, alpha=0.5, iters=5):
    key = (id(st), k, alpha, iters)
    if key not in _gcache: _gcache[key] = (smooth(P, graph_matrix(st["cand"], st["lo"], st["n"], k=k), alpha, iters), base_chains(st))
    return _gcache[key]

def make(lo, hi, ps=0.8, **g):
    def f(P, st):
        Pg, ch = Pg_of(P, st, **g); return calibrate_counts(Pg, ch, lo=lo, hi=hi, p_stay=ps)[0]
    return f

dec = {"base": make(60, 160)}
for lo in (75, 85, 90):
    for hi in (130, 160):
        dec[f"c{lo}_{hi}"] = make(lo, hi)
dec["c85_160_ps.9"] = make(85, 160, 0.9); dec["c85_160_ps.7"] = make(85, 160, 0.7)
run(dec, tag="calibration sweep")
