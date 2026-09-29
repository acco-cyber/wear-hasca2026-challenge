import numpy as np
from common import *
d = load_oof(); y, sbj = d["y"], d["sbj"]
pred = finish(d["P"], dict(sbj=sbj, sets=TRAIN_SETS)); o = np.load("oof_icm_labels.npy")
ch = o != pred
for nm, m in (("from-null", ch & (pred == 0)), ("to-null", ch & (o == 0)), ("act->act", ch & (pred > 0) & (o > 0))):
    print(f"{nm}: n {m.sum()} ({m.sum() / len(y):.4f}) new correct {np.mean(o[m] == y[m]):.3f} old correct {np.mean(pred[m] == y[m]):.3f}")
