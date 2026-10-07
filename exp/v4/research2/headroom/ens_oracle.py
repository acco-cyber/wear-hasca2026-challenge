"""How much diversity is left between the fits (upper bound for any fusion/selection): per-tile 'any fit right' oracle
and pairwise agreement, on refined OOF labels."""
import numpy as np
from hlib import *

D = setup(); y = D["y"]
F9 = np.load(os.path.join(K9, "stage.npz"), allow_pickle=True)
labs = {"K7": D["F"]["ref_oof"].astype(np.int64), "K9": F9["ref_oof"].astype(np.int64),
        "b4wa": np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64)}
for k, v in labs.items():
    log(f"{k}: {macro_f1(y, v):.4f}")
log(f"agreement K7/K9 {np.mean(labs['K7'] == labs['K9']):.4f}; both wrong {np.mean((labs['K7'] != y) & (labs['K9'] != y)):.4f}; "
    f"K7 errors {np.mean(labs['K7'] != y):.4f}, K9 errors {np.mean(labs['K9'] != y):.4f}")
orc = labs["b4wa"].copy(); m = (labs["K7"] == y) | (labs["K9"] == y); orc[m] = y[m]
log(f"oracle 'b4wa, or K7/K9 if either is right' {macro_f1(y, orc):.4f}")
orc = labs["K7"].copy(); m = labs["K9"] == y; orc[m] = y[m]
log(f"oracle 'K7 or K9 right' {macro_f1(y, orc):.4f}")
# disagreement-only oracle: where K7 != K9, take the right one (if any); elsewhere keep agreement
dis = labs["K7"] != labs["K9"]
log(f"tiles where K7 and K9 disagree {dis.sum()} ({dis.mean():.4f}); errors where they agree {np.sum(~dis & (labs['K7'] != y))}")
