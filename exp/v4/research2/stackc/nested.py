"""Strictly nested evaluation of stage C: for every outer subject fold k the decision rule (model variant x
light-quantile x raw/Sinkhorn, or "no change") is chosen by an INNER 4-fold CV on the other folds only (inner models
never see fold k), then the outer model (trained on the 4 other folds) is applied to fold k with that rule.
  python nested.py  -> logs + cache/nested.npz (OOF labels of the nested procedure, test labels of the full-data choice)"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stack as S
from stack import macro_f1

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"full": 1.0, "low": 0.2}
RULES = ["none"] + [f"light{q}_{d}" for q in (0.05, 0.1, 0.2) for d in ("raw", "sink1")] + ["all_raw", "all_sink1"]


def train_pred(X, T, tr, te, cat, rounds=400):
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=0)
    m = lgb.train(prm, lgb.Dataset(X[tr], T[tr], categorical_feature=cat), rounds)
    return m.predict(X[te]) if not isinstance(te, tuple) else m.predict(te[0])


def apply_rule(rule, p, C, Q, lab, sbj, cr):
    if rule == "none":
        return lab.copy()
    P = S.to_P(C, p, Q)
    d = rule.split("_")[1]
    new = P.argmax(1) if d == "raw" else S.sink(P, sbj, S.targets_from_Q(Q, sbj), 1.0).argmax(1)
    if rule.startswith("all"):
        return new
    q = float(rule.split("_")[0][5:]); out = lab.copy(); m = cr <= q; out[m] = new[m]
    return out


def main():
    z = np.load(os.path.join(HERE, "cache", "feat.npz"))
    Co, Fo, Go, Ct, Ft, Gt = (z[k] for k in ("Co", "Fo", "Go", "Ct", "Ft", "Gt"))
    gn = list(z["gn"]); names = list(z["fn"]) + gn
    y, sbj, fold, tsbj = z["y"], z["sbj"], z["fold"], z["tsbj"]
    Qo, Qt, lo, lt = z["Qo"].astype(np.float64), z["Qt"].astype(np.float64), z["lo"], z["lt"]
    Xo, Xt = S.rows(Co, Fo, Go), S.rows(Ct, Ft, Gt)
    n, K = Co.shape; T = (Co == y[:, None]).reshape(-1).astype(int); rf = np.repeat(fold, K)
    cr_o, cr_t = Go[:, gn.index("conf_rank")], Gt[:, gn.index("conf_rank")]
    cat = [names.index("cls"), names.index("sensor")]
    final_o = lo.copy(); chosen = {}
    outer_p = {v: np.zeros(n * K) for v in VARIANTS}
    for k in range(5):
        others = [f for f in range(5) if f != k]
        inner_p = {v: np.zeros(n * K) for v in VARIANTS}
        for v, ql in VARIANTS.items():
            low = np.repeat(cr_o <= ql, K)
            for j in others:
                tr = np.isin(rf, [f for f in others if f != j]) & low; te = rf == j
                inner_p[v][te] = train_pred(Xo, T, tr, te, cat)
            tr = (rf != k) & low; te = rf == k
            outer_p[v][te] = train_pred(Xo, T, tr, te, cat)
        ii = np.isin(fold, others)
        sc = {}
        for v in VARIANTS:
            pv = inner_p[v].reshape(n, K)[ii].reshape(-1)
            for r in RULES:
                if r == "none" and v != "full":
                    continue
                lab = apply_rule(r, pv, Co[ii], Qo[ii], lo[ii], sbj[ii], cr_o[ii])
                sc[(v, r)] = macro_f1(y[ii], lab)
        best = max(sc, key=sc.get); chosen[k] = best
        base = sc[("full", "none")]
        S.log(f"outer fold {k}: inner-CV best {best} {sc[best]:.4f} (none {base:.4f}); top3 "
              + ", ".join(f"{a}/{b} {sc[(a, b)]:.4f}" for a, b in sorted(sc, key=sc.get, reverse=True)[:3]))
        jj = fold == k; v, r = best
        pk = outer_p[v].reshape(n, K)[jj].reshape(-1)
        final_o[jj] = apply_rule(r, pk, Co[jj], Qo[jj], lo[jj], sbj[jj], cr_o[jj])
        S.log(f"  fold {k}: refined {macro_f1(y[jj], lo[jj]):.4f} -> nested stage C {macro_f1(y[jj], final_o[jj]):.4f}")
    S.log(f"NESTED stage C OOF {macro_f1(y, final_o):.4f} vs refined b4wa {macro_f1(y, lo):.4f}; per fold "
          + " ".join(f"{macro_f1(y[fold == f], lo[fold == f]):.4f}->{macro_f1(y[fold == f], final_o[fold == f]):.4f}" for f in range(5)))
    # full-data choice for the test labels: the rule that is best on the (outer) OOF of each variant
    sc = {}
    for v in VARIANTS:
        for r in RULES:
            sc[(v, r)] = macro_f1(y, apply_rule(r, outer_p[v], Co, Qo, lo, sbj, cr_o))
    best = max(sc, key=sc.get); v, r = best
    S.log(f"full-data choice {best} (OOF {sc[best]:.4f}, optimistic); all: " + ", ".join(f"{a}/{b} {s:.4f}" for (a, b), s in sorted(sc.items(), key=lambda x: -x[1])))
    low = np.repeat(cr_o <= VARIANTS[v], K)
    pt = train_pred(Xo, T, low, (Xt,), cat)
    lab_t = apply_rule(r, pt, Ct, Qt, lt, tsbj, cr_t)
    S.log(f"test: changed {np.mean(lab_t != lt):.4f} of tiles vs b4wa labt")
    np.savez(os.path.join(HERE, "cache", "nested.npz"), final_o=final_o, lab_t=lab_t, outer_full=outer_p["full"], outer_low=outer_p["low"], pt=pt)


if __name__ == "__main__":
    main()
