"""Graph-edge purity under mrf4: lo' = logit(sigmoid(lo) * p_same^gamma) for every candidate pair (chains unchanged).
Eval sessions: gpure_model.pkl (fit on the 10 train sessions x 2 draws); extra sessions: fold model that did not see the session.
python mrf_eval2.py [gammas] [variant]"""
import sys as _s, time
from lk import *
from gpure import fit_graph
from purity2 import load_train_structs, TRAIN_SESS
_s.path.insert(0, os.path.join(W, "exp", "decoder"))
from decoder import decode_subject, VARIANTS
from common import load_structs, sess_P, EXTRA

def cand_pure(P, st, gm, null_scale=0.5):
    n = len(P); Pn = P.copy(); Pn[:, 0] *= null_scale; Pn /= Pn.sum(1, keepdims=True)
    Pg = graph_smooth(Pn, build_graph(st["cand"], st["lo"], n, k=10), alpha=0.5, iters=5)
    valid = st["cand"] >= 0; A = np.repeat(np.arange(n)[:, None], st["cand"].shape[1], 1)[valid]; J = st["cand"][valid]
    Wt = 1 / (1 + np.exp(-np.asarray(st["lo"], np.float64)[valid]))
    lg = lambda x: np.log(np.clip(x, 1e-9, 1)); m = 0.5 * (Pg[A] + Pg[J])
    js = 0.5 * (Pg[A] * (lg(Pg[A]) - lg(m))).sum(1) + 0.5 * (Pg[J] * (lg(Pg[J]) - lg(m))).sum(1)
    F = np.stack([np.log(Wt / (1 - Wt + 1e-9) + 1e-9), (Pn[A] * Pn[J]).sum(1), (Pg[A] * Pg[J]).sum(1), js, Pn[A, 0], Pn[J, 0], Pg[A, 0], Pg[J, 0],
                  Pn[A].max(1), Pn[J].max(1), (Pg[A].argmax(1) == Pg[J].argmax(1)).astype(float)], 1).astype(np.float32)
    p = np.ones(st["cand"].shape); p[valid] = gm.predict_proba(F)[:, 1]
    return p

def reweight(st, p, gamma):
    lo = np.asarray(st["lo"], np.float64); w = 1 / (1 + np.exp(-lo)) * p ** gamma; w = np.clip(w, 1e-12, 1 - 1e-9)
    lo2 = np.where(st["cand"] >= 0, np.log(w / (1 - w)), lo).astype(np.float32)
    return dict(st, lo=lo2)

if __name__ == "__main__":
    gammas = [float(x) for x in (_s.argv[1] if len(_s.argv) > 1 else "1,2").split(",")]
    var = _s.argv[2] if len(_s.argv) > 2 else "mrf4"; cfg = VARIANTS[var]; t0 = time.time()
    oof = blend_oof(0.2); S = load_structs(("eval", "extra")); full = pickle.load(open(os.path.join(EXP, "gpure_model.pkl"), "rb"))
    TS = load_train_structs(); folds = [TRAIN_SESS[0::2], TRAIN_SESS[1::2]]
    fm = {k: fit_graph([st for s, _, st in TS if s in folds[1 - k]], oof) for k in range(2)}; print("fold models fit", f"({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(fm, open(os.path.join(EXP, "gpure_fold_models.pkl"), "wb"))
    rows = []
    for s in [x for x in EVAL + EXTRA if x in S]:
        st = S[s]; P = sess_P(oof, st); m = full if s in EVAL else (fm[0] if s in folds[0] else fm[1])
        p = cand_pure(P, st, m); r = dict(session=s, base=f1(st["y"], decode_subject(P, st, cfg, None)))
        for g in gammas: r[f"g{g}"] = f1(st["y"], decode_subject(P, reweight(st, p, g), cfg, None))
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).set_index("session"); ex = [s for s in df.index if s not in EVAL]
    df.loc["EVAL_MEAN"] = df.loc[[s for s in EVAL if s in df.index]].mean(); df.loc["EXTRA_MEAN"] = df.loc[ex].mean()
    print(df.round(4).to_string())
