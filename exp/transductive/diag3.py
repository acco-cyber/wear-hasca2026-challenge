"""Error structure of the (new) baseline decode: act->null, null->act, act->other act; per-class recall/precision."""
import os, pickle
from tlib import *
from sklearn.metrics import f1_score

C = pickle.load(open(os.path.join(TD, "cache_eval.pkl"), "rb")); C.update(pickle.load(open(os.path.join(TD, "cache_extra.pkl"), "rb")))
rows = []; CM = np.zeros((NC, NC))
for s, d in C.items():
    y = d["y"]; l = d["lab0"]; e = y != l
    CM += np.bincount(y * NC + l, minlength=NC * NC).reshape(NC, NC)
    rows.append(dict(s=s, f=round(d["f0"], 4), err=round(e.mean(), 3), act2null=round(((y > 0) & (l == 0)).mean(), 3),
                     null2act=round(((y == 0) & (l > 0)).mean(), 3), act2act=round(((y > 0) & (l > 0) & e).mean(), 3),
                     null_true=round((y == 0).mean(), 3), null_pred=round((l == 0).mean(), 3)))
df = pd.DataFrame(rows); print(df.to_string(index=False)); print(df.mean(numeric_only=True).round(3).to_dict())
rec = np.diag(CM) / CM.sum(1); prec = np.diag(CM) / CM.sum(0)
print("class  recall  precision  ->null  top_confusion")
for c in range(NC):
    r = CM[c].copy(); r[c] = 0; j = r.argmax()
    print(f"{c:5d}  {rec[c]:.3f}  {prec[c]:.3f}  {CM[c,0]/CM[c].sum():.3f}  {j}:{r[j]/CM[c].sum():.3f}")
