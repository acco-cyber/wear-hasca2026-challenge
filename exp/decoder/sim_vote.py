"""Majority vote across decodes of several OOF blends (and seeds) on eval+extra sim sessions."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_structs, sess_P, mf1, EVAL, EXTRA
from decoder import decode_subject, VARIANTS
from sim_oof import load_spec

W = r"E:\Claude code\wear\work"; B = r"E:\Claude code\wear\exp\base"
SPECS = {"v1f": f"{W}\\lgbm_v1\\oof.npy:0.8,{W}\\fusion_v1\\oof.npy:0.2",
         "v3bf": f"{B}\\v3b\\oof.npy:0.8,{W}\\fusion_v1\\oof.npy:0.2",
         "v3bv1f": f"{B}\\v3b\\oof.npy:0.45,{W}\\lgbm_v1\\oof.npy:0.35,{W}\\fusion_v1\\oof.npy:0.2"}
S = load_structs(("eval", "extra")); oofs = {k: load_spec(v) for k, v in SPECS.items()}
res = {k: {} for k in ["v3bf_mix6", "vote3_mrf4", "vote_v3bf2_v1f1_mix6"]}
for s in [x for x in EVAL + EXTRA if x in S]:
    st = S[s]; y = st["y"]; labs = {}
    for k in SPECS:
        P = sess_P(oofs[k], st)
        labs[(k, "mrf4")] = decode_subject(P, st, VARIANTS["mrf4"], None)
        labs[(k, "mix6")] = decode_subject(P, st, VARIANTS["mix6"], None)
    def vote(items, w=None):
        C = np.zeros((len(y), 19))
        for i, it in enumerate(items):
            C[np.arange(len(y)), labs[it]] += (w[i] if w else 1.0) + 1e-3 * (len(items) - i)
        return C.argmax(1)
    res["v3bf_mix6"][s] = mf1(y, labs[("v3bf", "mix6")])
    res["vote3_mrf4"][s] = mf1(y, vote([("v3bf", "mrf4"), ("v3bv1f", "mrf4"), ("v1f", "mrf4")]))
    res["vote_v3bf2_v1f1_mix6"][s] = mf1(y, vote([("v3bf", "mix6"), ("v3bv1f", "mix6"), ("v1f", "mix6")], [2.0, 1.5, 1.0]))
    print(s, {k: round(v[s], 4) for k, v in res.items()}, flush=True)
for k, v in res.items():
    ev = np.mean([v[s] for s in EVAL if s in v]); ex = np.mean([v[s] for s in v if s not in EVAL])
    print(f"{k}: eval {ev:.4f} extra {ex:.4f}")
