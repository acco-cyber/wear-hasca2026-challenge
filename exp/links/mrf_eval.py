"""Evaluate the purity-cut links under the decoder agent's mrf4 (exp/decoder/decoder.py) on eval + extra sim sessions.
The cut is expressed in the standard struct format: sc of cut assignment edges -> -50 (so cut(.., thr=-6) drops them).
Eval sessions use purity2_model.pkl (fit on 10 train sessions x 2 draws). Extra sessions (sbj_2,8,13,17,4,11 are purity training sessions)
use the 2-fold model that did NOT see that session.
python mrf_eval.py [taus] [variant]"""
import sys as _s, time
from lk import *
from purity2 import edge_feats2, fit, load_train_structs, TRAIN_SESS
_s.path.insert(0, os.path.join(W, "exp", "decoder"))
from decoder import decode_subject, VARIANTS
from common import load_structs, sess_P, EXTRA

def cut_struct(st, P, model, tau):
    a, b, succ, F, Pg = edge_feats2(P, dict(st, Lm=np.asarray(st["Lm"], np.float32))); p = model.predict_proba(F)[:, 1]
    sc2 = np.array(st["sc"], np.float32).copy(); sc2[a[p < tau]] = -50.0
    return dict(st, sc=sc2), p

if __name__ == "__main__":
    taus = [float(x) for x in (_s.argv[1] if len(_s.argv) > 1 else "0.5,0.6").split(",")]
    var = _s.argv[2] if len(_s.argv) > 2 else "mrf4"; cfg = VARIANTS[var]; t0 = time.time()
    oof = blend_oof(0.2); S = load_structs(("eval", "extra"))
    full = pickle.load(open(os.path.join(EXP, "purity2_model.pkl"), "rb"))
    TS = load_train_structs(); folds = [TRAIN_SESS[0::2], TRAIN_SESS[1::2]]
    fm = {k: fit([st for s, _, st in TS if s in folds[1 - k]], oof) for k in range(2)}; print("fold models fit", f"({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for s in [x for x in EVAL + EXTRA if x in S]:
        st = S[s]; P = sess_P(oof, st)
        m = full if s in EVAL else (fm[0] if s in folds[0] else fm[1])
        r = dict(session=s, base=f1(st["y"], decode_subject(P, st, cfg, None)))
        for tau in taus:
            st2, p = cut_struct(st, P, m, tau); r[f"cut{tau}"] = f1(st["y"], decode_subject(P, st2, cfg, None))
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).set_index("session"); ex = [s for s in df.index if s not in EVAL]
    df.loc["EVAL_MEAN"] = df.loc[[s for s in EVAL if s in df.index]].mean(); df.loc["EXTRA_MEAN"] = df.loc[ex].mean()
    print(df.round(4).to_string())
