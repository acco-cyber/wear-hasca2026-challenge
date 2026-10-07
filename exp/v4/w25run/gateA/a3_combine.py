"""Gate A: v4_combine.main() unchanged, with two optional patches (no edit of v4_combine.py):
  --relink tag=links.npz[,tag=links.npz]  the full fit of part local:<tag>:<dir> carries these matchings (oof/test succ+score)
                                          instead of its keep4 own links, so the fused REFINER walks the chain links too
                                          (plain v4_combine always refines along keep4/links.npz)
  --outdir DIR                            write sub_v4c_<tag>* under DIR/subs instead of wear/subs
All other arguments are passed to v4_combine.
  python a3_combine.py [--relink ...] [--outdir DIR] --parts ... --onehot --tag T"""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
args = sys.argv[1:]; relink, outdir = {}, ""
if "--relink" in args:
    i = args.index("--relink"); relink = dict(kv.split("=", 1) for kv in args[i + 1].split(",")); del args[i:i + 2]
if "--outdir" in args:
    i = args.index("--outdir"); outdir = args[i + 1]; del args[i:i + 2]
import v4_combine as C
REAL_W = C.W
orig = C.load_part


def load_part(spec):
    C.W = REAL_W                                     # parts are read from the real subs folder
    f = orig(spec)
    if outdir:
        C.W = outdir
    if spec.startswith("local:"):
        tag = spec.split(":", 2)[1]
        if tag in relink:
            z = np.load(relink[tag])
            f.update(oof_succ=z["oof_succ"].astype(np.int64), oof_score=z["oof_score"].astype(np.float32),
                     test_succ=z["test_succ"].astype(np.int64), test_score=z["test_score"].astype(np.float32))
            ts = f["true_succ"].astype(np.int64); h = ts >= 0
            C.log(f"part {tag}: refiner links from {os.path.basename(relink[tag])}, exact successor member0 {(f['oof_succ'][0][h] == ts[h]).mean():.4f}")
    return f


C.load_part = load_part
if outdir:
    os.makedirs(os.path.join(outdir, "subs"), exist_ok=True); C.W = outdir
sys.argv = [os.path.join(os.path.dirname(C.__file__), "v4_combine.py")] + args
C.main()
