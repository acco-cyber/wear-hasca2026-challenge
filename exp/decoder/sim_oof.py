"""Evaluate decoder variants on eval+extra sim sessions for arbitrary OOF files.
python sim_oof.py --variant mrf4 --oofs "name=path[,path2:w2 ...];name2=..."
Each OOF spec: comma list of path:weight (log-space blend, weights normalised)."""
import os, sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load_structs, sess_P, mf1, EVAL, EXTRA
from decoder import decode_subject, VARIANTS

def load_spec(spec):
    L = 0; ws = 0
    for part in spec.split(","):
        p, w = (part.rsplit(":", 1) + ["1"])[:2] if ":" in part[2:] else (part, "1")
        w = float(w); X = np.load(p); L = L + w * np.log(np.clip(X, 1e-6, 1)); ws += w
    L = L / ws; P = np.exp(L); P /= np.nansum(P, 2, keepdims=True); return P

if __name__ != "__main__":
    pass
else:
  ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="mrf4"); ap.add_argument("--oofs", required=True)
  a = ap.parse_args(); S = load_structs(("eval", "extra")); rows = {}
  for item in a.oofs.split(";"):
    name, spec = item.split("=", 1); oof = load_spec(spec); cfg = VARIANTS[a.variant]
    r = {}
    for s in [x for x in EVAL + EXTRA if x in S]:
        st = S[s]; r[s] = mf1(st["y"], decode_subject(sess_P(oof, st), st, cfg, None))
    ev = np.mean([r[s] for s in EVAL if s in r]); ex = np.mean([r[s] for s in r if s not in EVAL])
    r["EVAL"] = ev; r["EXTRA"] = ex; rows[name] = r
    print(f"{name}: eval {ev:.4f} extra {ex:.4f}", {k: round(v, 4) for k, v in r.items()}, flush=True)
