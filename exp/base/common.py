"""Shared helpers for the base-classifier experiments (exp/base)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep"); WORK = r"E:\Claude code\wear\work"
EXP = r"E:\Claude code\wear\exp\base"
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
NC = 19
FAMILY = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7])

_meta = None
def meta():
    global _meta
    if _meta is None: _meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    return _meta

def folds_by_sbj(nfold=5):
    m = meta(); subjects = np.unique(m.sbj); perm = np.random.RandomState(0).permutation(subjects)
    return {s: i % nfold for i, s in enumerate(perm)}

def sec_fold():
    fo = folds_by_sbj(); return np.array([fo[s] for s in meta().sbj.to_numpy()])

def single_limb_pick(oof, seed=5):
    valid = ~np.isnan(oof[:, :, 0]); rng = np.random.RandomState(seed)
    rl = np.array([rng.choice(np.where(v)[0]) if v.any() else -1 for v in valid])
    return rl

_rl = None
def eval_single(oof, seed=5, verbose=True, name=""):
    """Macro-F1 with one random valid limb per second (seed 5), seconds with purity>=0.8 and >=1 valid limb."""
    global _rl
    m = meta(); y = m.y.to_numpy(); pur = m.pur.to_numpy()
    if _rl is None:
        # fixed pick: valid = limb has IMU (not NaN in the reference lgbm_v1 oof) and purity >= 0.8
        ref = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy"), mmap_mode="r")
        _rl = single_limb_pick(np.asarray(ref[:, :, :1]), seed)
    ok = (_rl >= 0) & (pur >= 0.8)
    idx = np.where(ok)[0]
    P = oof[idx, _rl[idx]]
    bad = np.isnan(P[:, 0]); P = np.where(bad[:, None], 1.0 / NC, P)
    f = f1_score(y[idx], P.argmax(1), average="macro")
    if verbose: print(f"[{name}] OOF single-limb macro-F1 {f:.4f} (n={len(idx)}, nan rows {bad.sum()})", flush=True)
    return f

def to_log(P):
    return np.log(np.clip(P, 1e-6, 1))

def norm_probs(L):
    """log-scores -> probs along last axis (NaN-preserving)."""
    L = L - np.nanmax(L, -1, keepdims=True); P = np.exp(L); return P / np.nansum(P, -1, keepdims=True)

def blend(oofs, ws):
    L = sum(w * to_log(o) for o, w in zip(oofs, ws)); return norm_probs(L).astype(np.float32)

def run_sim(oof_path, tag=""):
    import subprocess, re, sys
    env = dict(os.environ); env["OMP_NUM_THREADS"] = "4"
    r = subprocess.run([sys.executable, r"E:\Claude code\wear\src\sim_decode.py", "--oof", oof_path], capture_output=True, text=True, env=env,
                       cwd=r"E:\Claude code\wear\src")
    out = r.stdout
    m = re.search(r"MEAN: (\{.*\})", out)
    res = eval(m.group(1)) if m else None
    with open(os.path.join(EXP, "sim_log.txt"), "a") as f:
        f.write(f"==== {tag} {oof_path}\n{out}\n{r.stderr[-2000:] if r.returncode else ''}\n")
    return res, out
