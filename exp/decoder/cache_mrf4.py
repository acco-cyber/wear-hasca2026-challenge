"""Run mrf4 on all sim sessions and cache labels + intermediate quantities -> mrf4_cache.pkl"""
import pickle, numpy as np
from common import load_oof, load_structs, sess_P, mf1, EVAL, EXTRA
from decoder import decode_subject, VARIANTS
oof = load_oof(0.2); S = load_structs(("eval", "extra")); out = {}
for s in [x for x in EVAL + EXTRA if x in S]:
    st = S[s]; P = sess_P(oof, st)
    lab, info = decode_subject(P, st, VARIANTS["mrf4"], return_info=True)
    out[s] = dict(lab=lab, logP=info["logP"], b=info["bias"], W=info["W"], P=P, y=st["y"], a=st["a"], b_=st["b"])
    print(s, round(mf1(st["y"], lab), 4), flush=True)
pickle.dump(out, open("mrf4_cache.pkl", "wb"))
