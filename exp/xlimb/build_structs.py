"""Rebuild sim structs (same limbs + same video candidates as the stored structs) with an augmented scorer.
python build_structs.py <tag> [sets=eval,extra,extra2]  -> exp/xlimb/struct_<tag>_<set>.pkl"""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
from sklearn.metrics import roc_auc_score

def logodds(sc, F):
    m, cols = sc["m"], sc["cols"]; sh = F.shape[:-1]; Z = F.reshape(-1, F.shape[-1])[:, cols]; out = np.full(len(Z), -50.0, np.float32)
    ok = ~np.isnan(Z[:, 0]); p = np.clip(m.predict_proba(Z[ok])[:, 1], 1e-6, 1 - 1e-6); out[ok] = np.log(p / (1 - p))
    return out.reshape(sh)

def load_old(w):
    return pickle.load(open(os.path.join(WORK, "sim_struct.pkl") if w == "eval" else os.path.join(TD, f"{w}_struct.pkl"), "rb"))

def _job(args):
    tag, w, s = args
    t0 = time.time(); sc = pickle.load(open(os.path.join(XD, f"scorer_{tag}.pkl"), "rb")); sc["m"].set_params(n_jobs=1)
    meta, imu, vid, sl = load_prep(); old = load_old(w)[s]; a, b, n = old["a"], old["b"], old["n"]; limb = np.asarray(old["limb"])
    V = np.asarray(vid[a:b], np.float32); W = session_windows(imu, a, b, limb)
    cand, F = all_pair_features(V, W, limb, sc["xl"])
    assert (cand == old["cand"]).all(), "candidate mismatch"
    lo = logodds(sc, F); succ0, scs, Lm = assignment(cand, lo, n)
    y = old["y"]; lab = (cand == (np.arange(n)[:, None] + 1)); v = cand >= 0
    same = np.where(v, y[np.where(v, cand, 0)] == y[:, None], False)
    met = {}
    for nm, L, su, sc_ in (("old", old["lo"], old["succ0"], old["sc"]), ("new", lo, succ0, scs)):
        e = sc_ >= -6.0; tr = su == np.arange(n) + 1
        met[nm] = dict(auc=roc_auc_score(lab[v], L[v]), auc_same=roc_auc_score(same[v], L[v]), prec=float(tr[e].mean()), n_e=int(e.sum()),
                       same_lab=float((y[e] == y[su[e]]).mean()))
    st = dict(a=a, b=b, n=n, y=y, limb=limb, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm.astype(np.float16))
    return s, st, met, time.time() - t0

if __name__ == "__main__":
    tag = sys.argv[1]; sets = (sys.argv[2] if len(sys.argv) > 2 else "eval,extra,extra2").split(",")
    jobs = [(tag, w, s) for w in sets for s in SESS[w]]; out = {w: {} for w in sets}; M = {}
    with ProcessPoolExecutor(3) as ex:
        for (tg, w, s), (s_, st, met, dt) in zip(jobs, ex.map(_job, jobs)):
            out[w][s] = st; M[s] = met
            print(f"{w} {s} ({dt:.0f}s) " + " | ".join(f"{k}: " + " ".join(f"{kk} {vv:.3f}" for kk, vv in d.items()) for k, d in met.items()), flush=True)
    for w in sets:
        pickle.dump(out[w], open(os.path.join(XD, f"struct_{tag}_{w}.pkl"), "wb"))
        for k in ("old", "new"):
            print(w, k, " ".join(f"{m} {np.mean([M[s][k][m] for s in SESS[w]]):.4f}" for m in ("auc", "auc_same", "prec", "same_lab", "n_e")))
