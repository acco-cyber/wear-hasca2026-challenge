"""Log-linear blends of base models: writes exp/base/<name>/{oof,test}.npy and prints OOF single-limb F1.
python blend_eval.py name=m1:w1+m2:w2 [name2=...]   (model: exp/base dir name, or w:<work dir name>)"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def mdir(m): return os.path.join(WORK, m[2:]) if m.startswith("w:") else os.path.join(EXP, m)

cache = {}
def load(m):
    if m not in cache: cache[m] = (np.load(os.path.join(mdir(m), "oof.npy")), np.load(os.path.join(mdir(m), "test.npy")))
    return cache[m]

for spec in sys.argv[1:]:
    name, rest = spec.split("=")
    parts = [p.rsplit(":", 1) if not p.startswith("w:") or p.count(":") > 1 else (p, "1") for p in rest.split("+")]
    ms = [p[0] for p in parts]; ws = [float(p[1]) for p in parts]
    O = blend([load(m)[0] for m in ms], ws); T = blend([load(m)[1] for m in ms], ws)
    od = os.path.join(EXP, name); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), O); np.save(os.path.join(od, "test.npy"), T)
    eval_single(O, name=f"{name} = {rest}")
