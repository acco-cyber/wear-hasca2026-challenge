"""Cache per-session, per-limb descriptors for all train sessions -> cache/<session>.npz"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import load_session, TRAIN_DIR
from feats import describe

OUT = os.path.join(r"E:\Claude code\wear\exp\hyb\agents\group", "cache")
os.makedirs(OUT, exist_ok=True)
names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(TRAIN_DIR, "sbj_*.csv")))]
for name in names:
    f = os.path.join(OUT, name + ".npz")
    if os.path.exists(f):
        continue
    t0 = time.time()
    acc, lab = load_session(name)
    D = [describe(acc[l]) for l in range(4)]
    np.savez(f, lab=lab.astype(str),
             **{f"{k}{l}": D[l][k] for l in range(4) for k in ("desc", "z_spec", "z_specj", "z_acf", "std", "dyn", "gmean")})
    print(name, acc.shape[1], "s", round(time.time() - t0, 1), "s", flush=True)
print("done")
