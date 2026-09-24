"""v2a (encoder) links vs baseline links under mrf4, eval sessions only."""
import sys as _s
from lk import *
_s.path.insert(0, os.path.join(W, "exp", "decoder"))
from decoder import decode_subject, VARIANTS
from common import load_structs, sess_P
oof = blend_oof(0.2); S0 = load_structs(("eval",)); S1 = pickle.load(open(os.path.join(EXP, "sim_struct_v2a.pkl"), "rb")); cfg = VARIANTS["mrf4"]; r0, r1 = [], []
for s in EVAL:
    P = sess_P(oof, S0[s]); r0.append(f1(S0[s]["y"], decode_subject(P, S0[s], cfg, None))); r1.append(f1(S1[s]["y"], decode_subject(P, S1[s], cfg, None)))
    print(s, round(r0[-1], 4), round(r1[-1], 4), flush=True)
print("mrf4 baseline links", round(np.mean(r0), 4), "| v2a links", round(np.mean(r1), 4))
