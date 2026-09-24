"""Average successor log-odds of several scorers.  python ens_preds.py <out_tag> <tag1+tag2+...>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import SESS, XD
PD = os.path.join(XD, "preds")
out, tags = sys.argv[1], sys.argv[2].split("+")
for w in SESS:
    for s in SESS[w]:
        L = [np.load(os.path.join(PD, f"{t}_{s}.npz")) for t in tags]
        np.savez(os.path.join(PD, f"{out}_{s}.npz"), lo=np.mean([d["lo"] for d in L], 0).astype(np.float32), lq=np.mean([d["lq"] for d in L], 0).astype(np.float32))
print("ok")
