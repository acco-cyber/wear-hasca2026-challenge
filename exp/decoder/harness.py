"""Evaluate a dict of decoders on eval (+ extra) sessions. Each decoder: fn(P, st) -> labels."""
import time, numpy as np, pandas as pd
from common import *

_CACHE = {}
def data(which=("eval", "extra"), w2=0.2):
    key = (tuple(which), w2)
    if key not in _CACHE: _CACHE[key] = (load_oof(w2), load_structs(which))
    return _CACHE[key]

def run(decoders, sessions=None, which=("eval", "extra"), w2=0.2, verbose=True, tag=""):
    oof, S = data(which, w2)
    if sessions is None: sessions = [s for s in EVAL + EXTRA if s in S]
    rows = []
    for s in sessions:
        st = S[s]; P = sess_P(oof, st); y = st["y"]; r = dict(session=s)
        for name, fn in decoders.items():
            t0 = time.time(); lab = fn(P, st); r[name] = mf1(y, lab)
        rows.append(r)
        if verbose: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows).set_index("session")
    ev = [s for s in sessions if s in EVAL]; ex = [s for s in sessions if s not in EVAL]
    summ = pd.DataFrame({"EVAL_MEAN": df.loc[ev].mean(), "EXTRA_MEAN": df.loc[ex].mean() if ex else np.nan})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(tag); print(df.round(4).to_string()); print(summ.T.round(4).to_string(), flush=True)
    return df
