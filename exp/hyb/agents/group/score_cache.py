"""Compute and cache the 6 limb-pair log-odds matrices per session (true second order) -> scores/<model>_<session>.npz
usage: python score_cache.py <model1[,model2]> <session> [...]   (session may also be test:22 etc.)"""
import sys, os, time, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
import lightgbm as lgb
from feats import score_matrix, describe
from common import PAIRS, PAIR_ID

HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
CACHE = os.path.join(HERE, "cache"); OUT = os.path.join(HERE, "scores"); os.makedirs(OUT, exist_ok=True)
models = sys.argv[1].split(","); sessions = sys.argv[2:]
boosters = {m: lgb.Booster(model_file=os.path.join(HERE, m + ".txt")) for m in models}


def load_desc(name):
    if name.startswith("test:"):
        sb = int(name.split(":")[1])
        w = np.load(r"E:\Claude code\wear\work\w25\w25.npz")
        rows = [np.where((w["sbj"] == sb) & (w["limb"] == l))[0] for l in range(4)]
        return [describe(w["acc"][r]) for r in rows]
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    return [{k: z[f"{k}{l}"] for k in ("desc", "z_spec", "z_specj", "z_acf", "std", "dyn", "gmean")} for l in range(4)]


for name in sessions:
    D = load_desc(name)
    for m in models:
        f = os.path.join(OUT, f"{m}_{name.replace(':', '')}.npz")
        if os.path.exists(f):
            print("exists", f, flush=True); continue
        t0 = time.time(); S = {}
        for (a, b) in PAIRS:
            S[f"S{a}{b}"] = score_matrix(boosters[m], D[a], D[b], PAIR_ID[(a, b)]).astype(np.float16)
        np.savez(f, **S)
        print(f"{m} {name} N={len(D[0]['std'])} {time.time()-t0:.0f}s", flush=True)
print("done")
