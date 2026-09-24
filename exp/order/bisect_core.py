"""Joint variant assignment for 2-variant families (push-ups 11/12, sit-ups 13/14, lunges 16/17) by balanced spectral
bisection of the family's decoded windows on the link-scorer graph (protocol: each variant = its own bout-group(s),
normal/complex durations ~equal within a subject), parts -> variants by summed log-scores (joint assignment)."""
import numpy as np, scipy.sparse as sp, sys
DEC = r"E:\Claude code\wear\exp\decoder"
if DEC not in sys.path: sys.path.insert(0, DEC)
from common import graph_matrix
NC = 19
PAIRS = {"push": (11, 12), "sit": (13, 14), "lun": (16, 17)}

def fiedler(W, idx):
    Ws = W[idx][:, idx]; dg = np.asarray(Ws.sum(1)).ravel() + 1e-9; Dm = sp.diags(1 / np.sqrt(dg))
    vals, vecs = np.linalg.eigh((Dm @ Ws @ Dm).toarray()); o = np.argsort(-vals)
    return vecs[:, o[1]] / np.sqrt(dg), vals[o[1]]

def bisect_family(W, L, lab, cls):
    """Returns (idx, s) where s in (0,1) = soft membership of cls[1] for windows idx (decoded in family), or None."""
    idx = np.where(np.isin(lab, cls))[0]
    if len(idx) < 20: return None
    v, lam2 = fiedler(W, idx); z = (v - np.median(v)) / (np.std(v) + 1e-12); part = z > 0
    sA = L[idx][part][:, list(cls)].sum(0); sB = L[idx][~part][:, list(cls)].sum(0)
    if (sA[1] - sA[0]) < (sB[1] - sB[0]): z = -z          # orient: positive side -> cls[1]
    return idx, z, lam2

def apply(L, lab, st, cfg, W=None):
    """cfg: fams, mode hard|soft, beta, gamma. Returns new labels (hard) or new log-scores (soft) and diagnostics."""
    n = len(lab); W = W if W is not None else _W(st, n)
    L2 = L.copy(); lab2 = lab.copy(); info = {}
    for f in cfg["fams"]:
        cls = PAIRS[f]; r = bisect_family(W, L, lab, cls)
        if r is None: continue
        idx, z, lam2 = r; info[f] = lam2
        if cfg["mode"] == "hard":
            lab2[idx] = np.where(z > 0, cls[1], cls[0])
        else:
            s = 1 / (1 + np.exp(-cfg["gamma"] * z)); s = np.clip(s, 0.02, 0.98)
            L2[idx, cls[1]] += cfg["beta"] * np.log(s); L2[idx, cls[0]] += cfg["beta"] * np.log(1 - s)
    return lab2, L2, info

def _W(st, n):
    W = graph_matrix(st["cand"], st["lo"], n, k=10); return (W + W.T).tocsr()
