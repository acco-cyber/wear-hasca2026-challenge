"""OOF (5-fold by subject) + test count predictions for a fixed method = average of (feature set, regressor) members.
python make_counts.py <name> <FS:model>[,<FS:model>...]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clib as C
from clib import FOLDS

FS = {
    "F0": dict(mo=["Po", "Bpo"]),
    "F1": dict(mo=["Po", "Bpo", "P7o", "P9o", "B7o", "B9o"]),
    "F4": dict(mo=["Po", "Bpo", "Q7o", "Q9o"]),
    "F7": dict(mo=["Po", "Bpo", "Q7o", "Q9o", "P7o", "P9o", "B7o", "B9o"]),
    "E4": dict(mo=["Po", "Bpo", "Q7o", "Q9o"], ext=True),
}
_built = {}


def built(D, nm):
    if nm not in _built:
        o = FS[nm]
        _built[nm] = C.build(D, [D[k] for k in o["mo"]], [D[k[:-1] + "t"] for k in o["mo"]], ext=o.get("ext", False))
    return _built[nm]


def member(D, fs, mdl):
    X, key, Xt, kt, true, kf = built(D, fs)
    cnt = C.cv_predict(C.MODELS[mdl], X, true, key, kf, range(FOLDS), range(FOLDS))
    ct = C.MODELS[mdl](X, true, key, Xt, kt)
    return cnt, ct


if __name__ == "__main__":
    name, spec = sys.argv[1], sys.argv[2]
    D = C.load(); mem = [s.split(":") for s in spec.split(",")]
    for fs, _ in mem:
        built(D, fs)
    res = Parallel(n_jobs=min(3, len(mem)))(delayed(member)(D, fs, m) for fs, m in mem)
    cnt = np.mean([r[0] for r in res], 0); ct = np.mean([r[1] for r in res], 0)
    X, key, Xt, kt, true, kf = built(D, mem[0][0])
    f, _ = C.f1_from_counts(D, D["Po"], key, cnt)
    print(f"{name}: err {np.abs(cnt - true).mean():.2f} F1 {f:.4f}; test mean {ct.mean():.1f}")
    np.savez(os.path.join(C.HERE, f"counts_{name}.npz"), cnt=cnt, ct=ct, key=key, kt=kt, true=true)
