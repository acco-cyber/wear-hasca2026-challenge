"""Lever 2/3: successor links. K7 single fit, OOF only, learned nested counts unless stated.
  python run_links.py true      -> 8 x true successor chain (score = p90 of the kernel's link scores), + with true counts
  python run_links.py part      -> 50% / 80% of every member's wrong links replaced by the true successor"""
import sys
import numpy as np
from hlib import *

mode = sys.argv[1]
D = setup(); y = D["y"]
res = {}
if mode == "true":
    L = true_links(D)
    r = decode(D, L, "learned")
    res["true_links"] = (macro_f1(y, r["fin"]), macro_f1(y, refine_oof(D, r["fin"], L, r["P"], r["Q"])))
    np.savez_compressed(os.path.join(HERE, "k7_truelinks.npz"), P0=r["P0"].astype(np.float32), P=r["P"].astype(np.float32), Q=r["Q"].astype(np.float32), fin=r["fin"])
    r2 = decode(D, L, "true", P0=r["P0"])
    res["true_links+true_counts"] = (macro_f1(y, r2["fin"]), macro_f1(y, refine_oof(D, r2["fin"], L, r2["P"], r2["Q"])))
    # the kernel links PLUS the true chain as a 9th member (does adding the truth to the bag help as much?)
    L9 = D["Lo"] + L[:1]
    r3 = decode(D, L9, "learned")
    res["kernel8+true1"] = (macro_f1(y, r3["fin"]), macro_f1(y, refine_oof(D, r3["fin"], L9, r3["P"], r3["Q"])))
elif mode == "typed":
    for kind in sys.argv[2].split(","):
        L = typed_links(D, kind)
        r = decode(D, L, "learned"); ref = refine_oof(D, r["fin"], L, r["P"], r["Q"])
        res[f"typed-{kind}"] = (macro_f1(y, r["fin"]), macro_f1(y, ref))
        np.savez_compressed(os.path.join(HERE, f"k7_links_typed_{kind}.npz"), fin=r["fin"], ref=ref)
else:
    fracs = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5, 0.8]
    for frac in fracs:
        L = corrected_links(D, frac)
        r = decode(D, L, "learned"); ref = refine_oof(D, r["fin"], L, r["P"], r["Q"])
        res[f"corrected{frac}"] = (macro_f1(y, r["fin"]), macro_f1(y, ref))
        np.savez_compressed(os.path.join(HERE, f"k7_links_corr{frac}.npz"), fin=r["fin"], ref=ref)
for k, v in res.items():
    log(f"RESULT links {k:24s} pre-refiner {v[0]:.4f} refined {v[1]:.4f}")
