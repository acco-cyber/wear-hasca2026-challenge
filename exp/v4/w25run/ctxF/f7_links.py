"""simulated chain links (research2/w25feas/cache/links_lgb_a1.npz, OOF only) + the fit's own test links as a placeholder
-> cache/links_lgb_a1_<fit>test.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
for nm, d, src in (("k7", K7, "links_lgb_a1.npz"), ("k9", K9, "links_lgb_a1_k9.npz")):
    a = np.load(os.path.join(FEAS, "cache", src)); b = np.load(os.path.join(d, "links.npz"))
    print(nm, {k: a[k].shape for k in a.files}, {k: b[k].shape for k in b.files})
    np.savez(os.path.join(CACHE, f"links_lgb_a1_{nm}test.npz"), oof_succ=a["oof_succ"], oof_score=a["oof_score"], test_succ=b["test_succ"], test_score=b["test_score"])
