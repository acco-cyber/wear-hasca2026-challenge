"""At label changes along matching 0 (fused decode), how often does the boundary pair equal the true bout pair around
the tile in the true order -- direct (both sides non-null) vs null-resolved through the chain."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_tr import chain_context
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "cache.npz"))
y, fin, ts = z["y"], z["fin_o"], z["true_succ"]
pbt, nbt, _, _ = chain_context(y, ts)          # true bout context in the true order
for nm, lab in (("decoded", fin), ("true labels", y)):
    succ = z["oof_succ"][0]
    pb, nb, rl, bl = chain_context(lab, succ)
    ii = np.flatnonzero((succ >= 0) & (lab != lab[np.maximum(succ, 0)])); jj = succ[ii]
    A, B = lab[ii], lab[jj]
    exact = jj == ts[ii]
    d = (A > 0) & (B > 0)
    print(f"{nm}: {len(ii)} label changes; direct act|act {d.mean():.3f} (link exact {exact[d].mean():.3f}); A|0 or 0|B {(~d).mean():.3f} (link exact {exact[~d].mean():.3f})")
    # truth check for direct pairs: true pair at i = (y[i], next true bout of i)
    tA = np.where(y[ii] > 0, y[ii], pbt[ii]); tB = nbt[ii]
    print(f"   direct pairs: A==true bout at i {(A[d] == y[ii][d]).mean():.3f}; (A,B)==(true bout, true next bout) {((A == tA) & (B == tB))[d].mean():.3f}")
    An = np.where(A > 0, A, pb[ii]); Bn = np.where(B > 0, B, nb[jj])
    m = ~d
    print(f"   resolved pairs: (An,Bn)==true pair {((An == tA) & (Bn == tB))[m].mean():.3f}, An==true {(An == tA)[m].mean():.3f}, Bn==true next {(Bn == tB)[m].mean():.3f}")
