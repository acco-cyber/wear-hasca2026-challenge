"""Stage C: per (tile, candidate) binary LightGBM stacker on top of the fused b4wa decode (+ refiner), nested by
subject fold; then Sinkhorn re-calibration to the fused count targets (= per-subject column sums of the fused Q).
  python stack.py --tag NAME [--extra cache/self.npz] [--drop f1,f2] [--train_low 0.2] [--rounds 400]"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import lightgbm as lgb
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1
from graph_lab import calibrate_targets

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
NC = 19


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def rows(C, F, G, extra=None):
    n, K, f = F.shape
    X = [F.reshape(n * K, f), np.repeat(G, K, 0)]
    if extra is not None:                       # extra per-class matrix (n,19) -> value at the candidate
        for E in extra:
            X.append(np.take_along_axis(E, C, 1).reshape(-1, 1))
    return np.concatenate(X, 1).astype(np.float32)


def to_P(C, p, Q, eps=0.05):
    """candidate scores -> (n,19) distribution: candidates share 1-eps*mass(non-candidates) ..."""
    n, K = C.shape
    P = Q * eps                                 # non-candidates keep a small multiple of the fused Q
    np.put_along_axis(P, C, 0.0, 1)
    pc = p.reshape(n, K); pc = pc / np.maximum(pc.sum(1, keepdims=True), 1e-12)
    np.put_along_axis(P, C, pc * (1 - P.sum(1))[:, None], 1)
    return P / P.sum(1, keepdims=True)


def targets_from_Q(Q, sbj):
    return {int(s): Q[sbj == s].sum(0) for s in np.unique(sbj)}


def sink(P, sbj, tg, a):
    R = np.clip(P, 1e-12, None) ** a; R /= R.sum(1, keepdims=True)
    return calibrate_targets(R, sbj, tg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--extra", default="")
    ap.add_argument("--drop", default=""); ap.add_argument("--train_low", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=31); ap.add_argument("--min_leaf", type=int, default=100)
    a = ap.parse_args()
    z = np.load(os.path.join(HERE, "cache", "feat.npz"))
    Co, Fo, Go, Ct, Ft, Gt = (z[k] for k in ("Co", "Fo", "Go", "Ct", "Ft", "Gt"))
    fn, gn = list(z["fn"]), list(z["gn"])
    y, sbj, fold, tsbj = z["y"], z["sbj"], z["fold"], z["tsbj"]
    Qo, Qt, lo, lt = z["Qo"].astype(np.float64), z["Qt"].astype(np.float64), z["lo"], z["lt"]
    exo = ext = None; en = []
    if a.extra:
        exo, ext = [], []
        for pth in a.extra.split(","):
            e = np.load(os.path.join(HERE, pth))
            for k in e.files:
                if k.endswith("_o"):
                    exo.append(e[k].astype(np.float32)); ext.append(e[k[:-2] + "_t"].astype(np.float32)); en.append(k[:-2])
    names = fn + gn + en
    Xo, Xt = rows(Co, Fo, Go, exo), rows(Ct, Ft, Gt, ext)
    keep = [i for i, nm in enumerate(names) if nm not in set(a.drop.split(","))]
    Xo, Xt, names = Xo[:, keep], Xt[:, keep], [names[i] for i in keep]
    n, K = Co.shape
    T = (Co == y[:, None]).reshape(-1).astype(int)
    rf = np.repeat(fold, K)
    cr_o = Go[:, gn.index("conf_rank")]; cr_t = Gt[:, gn.index("conf_rank")]
    trmask = np.repeat(cr_o <= a.train_low, K)
    cat = [names.index(c) for c in ("cls", "sensor") if c in names]
    prm = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=0)
    log(f"{Xo.shape[0]} rows x {Xo.shape[1]} features; positives {T.mean():.4f}; training rows {trmask.mean():.3f}")
    po = np.zeros(len(T))
    for k in range(5):
        tr = (rf != k) & trmask; te = rf == k
        m = lgb.train(prm, lgb.Dataset(Xo[tr], T[tr], categorical_feature=cat, free_raw_data=False), a.rounds)
        po[te] = m.predict(Xo[te])
        log(f"fold {k} done")
    m = lgb.train(prm, lgb.Dataset(Xo[trmask], T[trmask], categorical_feature=cat), a.rounds)
    pt = m.predict(Xt)
    imp = m.feature_importance("gain"); oi = np.argsort(-imp)
    log("top gain: " + ", ".join(f"{names[i]} {imp[i] / imp.sum():.3f}" for i in oi[:15]))
    PCo, PCt = to_P(Co, po, Qo), to_P(Ct, pt, Qt)
    tgo, tgt = targets_from_Q(Qo, sbj), targets_from_Q(Qt, tsbj)
    base_r, base_q = macro_f1(y, lo), macro_f1(y, Qo.argmax(1))
    pf = lambda lab: " ".join(f"{macro_f1(y[fold == f], lab[fold == f]):.4f}" for f in range(5))
    log(f"baseline: fused Qo argmax {base_q:.4f}, refined (b4wa labo) {base_r:.4f} | per fold {pf(lo)}")
    res = {}
    raw = PCo.argmax(1); res["raw"] = (raw, PCt.argmax(1))
    for aa in (1.0, 2.0):
        res[f"sink{aa:g}"] = (sink(PCo, sbj, tgo, aa).argmax(1), sink(PCt, tsbj, tgt, aa).argmax(1))
    # light: only the bottom q of per-subject confidence is re-judged
    for q in (0.05, 0.1, 0.2):
        for src in ("raw", "sink1"):
            lo2, lt2 = lo.copy(), lt.copy(); mo, mt = cr_o <= q, cr_t <= q
            lo2[mo] = res[src][0][mo]; lt2[mt] = res[src][1][mt]
            res[f"light{q:g}_{src}"] = (lo2, lt2)
    for nm, (lab_o, lab_t) in res.items():
        d = macro_f1(y, lab_o) - base_r
        log(f"{nm:>16}: OOF {macro_f1(y, lab_o):.4f} ({d:+.4f} vs refined) | per fold {pf(lab_o)} | changed OOF {np.mean(lab_o != lo):.4f}, test {np.mean(lab_t != lt):.4f}")
    np.savez(os.path.join(HERE, "cache", f"pred_{a.tag}.npz"), po=po.astype(np.float32), pt=pt.astype(np.float32),
             PCo=PCo.astype(np.float32), PCt=PCt.astype(np.float32), **{f"{k}_o": v[0] for k, v in res.items()}, **{f"{k}_t": v[1] for k, v in res.items()})


if __name__ == "__main__":
    main()
