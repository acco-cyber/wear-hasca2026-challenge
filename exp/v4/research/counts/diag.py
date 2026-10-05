"""Diagnostic grid (plain 5-fold subject CV, NOT used for selection of the reported model): feature sets x regressors."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import FOLDS, macro_f1

D = C.load(); y, fold = D["y"], D["fold"]
FS = {
    "F0": dict(mo=["Po", "Bpo"]),
    "F1": dict(mo=["Po", "Bpo", "P7o", "P9o", "B7o", "B9o"]),
    "F2": dict(mo=["Po", "Bpo"], subj=True),
    "F3": dict(mo=["Po", "Bpo", "P7o", "P9o", "B7o", "B9o"], subj=True),
    "F4": dict(mo=["Po", "Bpo", "Q7o", "Q9o"]),
    "F5": dict(mo=["Po", "Bpo"], center=True),
    "F6": dict(mo=["Po", "Bpo", "Q7o", "Q9o"], subj=True),
    "E0": dict(mo=["Po", "Bpo"], ext=True),
    "E4": dict(mo=["Po", "Bpo", "Q7o", "Q9o"], ext=True),
    "E1": dict(mo=["Po", "Bpo", "P7o", "P9o", "B7o", "B9o"], ext=True),
    "E7": dict(mo=["Po", "Bpo", "Q7o", "Q9o", "P7o", "P9o", "B7o", "B9o"], ext=True),
    "F7": dict(mo=["Po", "Bpo", "Q7o", "Q9o", "P7o", "P9o", "B7o", "B9o"]),
}
models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ridge", "ridge_log", "ridge_cen", "lgb", "avg"]
fsel = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FS)
built = {}
for nm in fsel:
    o = FS[nm]; mo = [D[k] for k in o["mo"]]; mt = [D[k[:-1] + "t"] for k in o["mo"]]
    built[nm] = C.build(D, mo, mt, subj=o.get("subj", False), center=o.get("center", False), ext=o.get("ext", False))


def run(fs, mdl):
    X, key, Xt, kt, true, kf = built[fs]
    cnt = C.cv_predict(C.MODELS[mdl], X, true, key, kf, range(FOLDS), range(FOLDS))
    return fs, mdl, cnt


t0 = time.time()
res = Parallel(n_jobs=3)(delayed(run)(fs, m) for fs in fsel for m in models)
for fs, mdl, cnt in res:
    X, key, Xt, kt, true, kf = built[fs]
    f, lab = C.f1_from_counts(D, D["Po"], key, cnt)
    np.save(os.path.join(C.HERE, f"diag_cnt_{fs}_{mdl}.npy"), cnt)
    print(f"{fs:3s} {mdl:9s} nfeat {X.shape[1]:3d}: err {np.abs(cnt - true).mean():5.2f} cen-err {C.centred_err(cnt, true, key):5.2f} F1 {f:.4f} | "
          + " ".join(f"{macro_f1(y[fold == k], lab[fold == k]):.4f}" for k in range(FOLDS)), flush=True)
print(f"done {time.time() - t0:.0f}s")
