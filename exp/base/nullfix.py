"""Add a log-bias to the null class so the OOF mean P(null) matches a reference model's (same bias applied to test).
python nullfix.py <src> <out> <ref>"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
def mdir(m): return os.path.join(WORK, m[2:]) if m.startswith("w:") else os.path.join(EXP, m)
src, out, ref = sys.argv[1:4]
O = np.load(os.path.join(mdir(src), "oof.npy")); T = np.load(os.path.join(mdir(src), "test.npy")); R = np.load(os.path.join(mdir(ref), "oof.npy"))
v = ~np.isnan(O[:, :, 0]) & ~np.isnan(R[:, :, 0]); target = R[v][:, 0].mean()
def shift(P, b):
    L = to_log(P); L[..., 0] += b; return norm_probs(L).astype(np.float32)
lo, hi = -3.0, 3.0
for _ in range(40):
    b = (lo + hi) / 2; mnull = shift(O[v], b)[:, 0].mean()
    if mnull < target: lo = b
    else: hi = b
print(f"{src}: null bias {b:.3f} -> mean P(null) {mnull:.3f} (target {target:.3f})")
od = os.path.join(EXP, out); os.makedirs(od, exist_ok=True)
np.save(os.path.join(od, "oof.npy"), shift(O, b)); np.save(os.path.join(od, "test.npy"), shift(T, b))
eval_single(np.load(os.path.join(od, "oof.npy")), name=out)
