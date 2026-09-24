"""Shared helpers for the 'order' experiments (protocol structure priors on top of the e7 recipe)."""
import os, sys, pickle
import numpy as np
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; PLD = r"E:\Claude code\wear\exp\pl"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (TD, DEC, PLD):
    if p not in sys.path: sys.path.insert(0, p)
NC = 19
SESS = {"eval": ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"],
        "extra": ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"],
        "extra2": ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]}
ALL = sum(SESS.values(), [])
OOF = r"E:\Claude code\wear\exp\base\bl_v3b_v1_f\oof.npy"
FAMS = {"jog": [1, 2, 3, 4, 5], "str": [6, 7, 8, 9, 10], "push": [11, 12], "sit": [13, 14], "lun": [16, 17]}
SINGLE = [15, 18]

def to_P(L):
    P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)

def dec(L, st, lo=None, hi=None):
    """mrf4 decode of log-scores L; optional per-class count bounds (arrays of length NC-1 or scalars)."""
    from decoder import decode_subject, VARIANTS
    cfg = dict(VARIANTS["mrf4"])
    if lo is not None: cfg["lo"] = lo
    if hi is not None: cfg["hi"] = hi
    return decode_subject(to_P(L), st, cfg, None)

def e7_stage(P, F, st, w=6.0):
    """e7 recipe. Returns L0 (base log-probs), L1 (after kNN label-spread term), lab0 (mrf4), lab1 (e7 labels), X."""
    from tlib import build_X
    from refine_core import knn_label_Q
    from decoder import decode_subject, VARIANTS
    L0 = np.log(np.clip(P, 1e-6, 1)); lab0 = decode_subject(P, st, VARIANTS["mrf4"], None)
    X = build_X(F, "v768", 128, 32); L1 = L0 + w * np.log(knn_label_Q(X, lab0, 5, 0.9, 0.1)); lab1 = dec(L1, st)
    return L0, L1, lab0, lab1, X

def family_bounds(lab, a=0.75, b=1.33, lo=80, hi=250, mode="fam", n_act_ref=None):
    """Per-class count bounds from the protocol's balanced-duration structure.
    mode 'fam': target m_f = mean decoded count over the variants of the family (family total is more reliable than the
    split); 'glob': m = decoded activity windows / 18 for every class; 'mix': geometric mean of both."""
    cnt = np.bincount(lab, minlength=NC).astype(float); lo_a = np.full(NC - 1, float(lo)); hi_a = np.full(NC - 1, float(hi))
    m_glob = cnt[1:].sum() / 18.0 if n_act_ref is None else n_act_ref / 18.0
    for f, cls in FAMS.items():
        m_f = cnt[cls].mean()
        m = m_f if mode == "fam" else (m_glob if mode == "glob" else np.sqrt(m_f * m_glob))
        for c in cls:
            lo_a[c - 1] = max(lo, a * m); hi_a[c - 1] = min(hi, max(b * m, lo_a[c - 1] + 10))
    if mode != "fam":
        for c in SINGLE:
            lo_a[c - 1] = max(lo, a * m_glob); hi_a[c - 1] = min(hi, max(b * m_glob, lo_a[c - 1] + 10))
    return lo_a, hi_a
