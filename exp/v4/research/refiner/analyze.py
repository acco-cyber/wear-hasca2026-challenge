"""Score cached refiner predictions: fixed threshold 0.5, leave-one-fold-out (LOFO) threshold / aggregation choice,
fully nested threshold when inner predictions exist, and LOFO selection among configs.
  python analyze.py cfg1 cfg2 ..."""
import os, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1
import rlib as R

GRID = np.round(np.arange(0.30, 0.71, 0.05), 2)


def f1s(y, lab, fold):
    return macro_f1(y, lab), [macro_f1(y[fold == f], lab[fold == f]) for f in range(R.FOLDS)]


def lab_for(fin, z, prob, thr, agg, rowmask=None):
    tl, ot = z["tl"], z["ot"]
    if rowmask is not None:
        tl, ot, prob = tl[rowmask], ot[rowmask], prob[rowmask]
    return R.flips(fin, tl, ot, prob, int(z["K"]), thr, agg)[0]


def lofo(fin, y, fold, z, prob_for_choice, choices):
    """for each fold k pick the choice maximising macro-F1 on folds != k (scored with prob_for_choice(k)), apply on k"""
    out = fin.copy(); picks = []
    Fr = fold[z["tl"]]
    for k in range(R.FOLDS):
        m = fold != k; best = None
        pk = prob_for_choice(k)
        for ch in choices:
            lab = lab_for(fin, z, pk, *ch, rowmask=Fr != k)
            s = macro_f1(y[m], lab[m])
            if best is None or s > best[0]:
                best = (s, ch)
        picks.append(best[1])
        lab = lab_for(fin, z, z["pr"], *best[1], rowmask=Fr == k); out[fold == k] = lab[fold == k]
    return out, picks


def main():
    d = R.load_cache(); y, fold, fin = d["y"], d["fold"], d["fin_o"]
    base = f1s(y, fin, fold)
    print(f"unrefined {base[0]:.4f} | " + " ".join(f"{v:.4f}" for v in base[1]))
    ref_base = None
    for cfg in sys.argv[1:]:
        p = os.path.join(R.HERE, "preds", f"{cfg}.npz")
        if not os.path.exists(p):
            print(cfg, "missing"); continue
        z = np.load(p)
        lab = lab_for(fin, z, z["pr"], 0.5, "sum"); s, pf = f1s(y, lab, fold)
        if ref_base is None and cfg == "base":
            ref_base = pf
        imp = sum(a > b for a, b in zip(pf, ref_base)) if ref_base else -1
        print(f"{cfg:16s} thr0.5 {s:.4f} | " + " ".join(f"{v:.4f}" for v in pf) + f" | folds>base {imp}")
        for agg in ("sum", "mean"):
            sc = [macro_f1(y, lab_for(fin, z, z["pr"], t, agg)) for t in GRID]
            print(f"   {agg} non-nested curve: " + " ".join(f"{t:.2f}:{v:.4f}" for t, v in zip(GRID, sc)))
        ch = [(t, "sum") for t in GRID]
        lo, picks = lofo(fin, y, fold, z, lambda k: z["pr"], ch); s, pf = f1s(y, lo, fold)
        print(f"   LOFO thr (sum)   {s:.4f} | " + " ".join(f"{v:.4f}" for v in pf) + f" | picks {[p_[0] for p_ in picks]}")
        ch2 = ch + [(t, "mean") for t in GRID]
        lo, picks = lofo(fin, y, fold, z, lambda k: z["pr"], ch2); s, pf = f1s(y, lo, fold)
        print(f"   LOFO thr+agg     {s:.4f} | " + " ".join(f"{v:.4f}" for v in pf) + f" | picks {picks}")
        if not np.isnan(z["inner"]).all():
            inn = z["inner"].astype(np.float64)
            lo, picks = lofo(fin, y, fold, z, lambda k: np.nan_to_num(inn[k]), ch); s, pf = f1s(y, lo, fold)
            print(f"   NESTED thr (sum) {s:.4f} | " + " ".join(f"{v:.4f}" for v in pf) + f" | picks {[p_[0] for p_ in picks]}")


if __name__ == "__main__":
    main()
