"""Honest tau selection: 2-fold CV over the 10 purity-training sessions (fit on 5, decode the other 5), new decoder."""
import time
from lk import *
from purity import fit_model, decode_pure, TRAIN_SESS
oof = blend_oof(0.2); ST = pickle.load(open(os.path.join(EXP, "purity_train_struct.pkl"), "rb")); t0 = time.time()
folds = [TRAIN_SESS[0::2], TRAIN_SESS[1::2]]; res = {}
for k in range(2):
    m = fit_model({s: ST[s] for s in folds[1 - k]}, oof)
    for s in folds[k]:
        st = ST[s]; st = dict(st, Lm=st["Lm"].astype(np.float32)); P = sim_P(oof, st); y = st["y"]
        for tau in (0.0, 0.3, 0.5, 0.6, 0.7):
            res.setdefault(tau, []).append(f1(y, decode_pure(P, st, m, tau)))
        print(s, {t: round(v[-1], 4) for t, v in res.items()}, f"({time.time()-t0:.0f}s)", flush=True)
for t, v in res.items(): print(f"tau {t}: mean {np.mean(v):.4f}")
