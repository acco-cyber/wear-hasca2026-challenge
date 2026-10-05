"""Second refiner iteration on the refined labels of a cached first-stage config.
  --mode stack : stage-1 labels = its outer-CV refined labels for every fold (usual stacking, mild leak)
  --mode nested: for outer fold k, stage-1 labels of folds != k come from the inner predictions (models without fold k),
                 fold k from the outer model; stage-2 trained on folds != k only -> fully nested
  python stage2.py --cfg fit_link_vote --mode nested [--thr2 0.5]"""
import os, sys, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp as E
import rlib as R
from v4_local import macro_f1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--mode", default="nested")
    ap.add_argument("--src", default="", help="preds name of stage 1 (default = cfg)"); ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args(); c = E.CFGS[a.cfg]; src = a.src or a.cfg
    d = R.load_cache(); y, fold, fin = d["y"], d["fold"], d["fin_o"]
    z = np.load(os.path.join(R.HERE, "preds", f"{src}.npz")); K = int(z["K"]); Fr1 = fold[z["tl"]]
    so = E.Side(d, "oof"); E.log("side ready")
    L1 = R.flips(fin, z["tl"], z["ot"], z["pr"], K)[0]
    E.log(f"stage 1 ({src}): {macro_f1(y, fin):.4f} -> {macro_f1(y, L1):.4f}")
    out = L1.copy(); pr2_all = []
    if a.mode == "stack":
        X, tl, ot, mid = E.build(so, L1, c["H"], c["feats"]); Fr = fold[tl]
        T = ((y[tl] == ot) & (y[tl] != L1[tl])).astype(int)
        E.log(f"stage 2 rows {X.shape}, {T.mean():.3f} should flip")
        pr, _ = E.cv(X, T, Fr, E.PARAMS, a.rounds, False)
        out = R.flips(L1, tl, ot, pr, K)[0]
        np.savez(os.path.join(R.HERE, "preds", f"{src}_s2stack.npz"), tl=tl, ot=ot, pr=pr, L1=L1, K=K)
    else:
        assert not np.isnan(z["inner"]).all(), "stage 1 needs --inner predictions"
        for k in range(R.FOLDS):
            inn = np.nan_to_num(z["inner"][k].astype(np.float64))
            Lk = R.flips(fin, z["tl"][Fr1 != k], z["ot"][Fr1 != k], inn[Fr1 != k], K)[0]
            Lk[fold == k] = L1[fold == k]
            X, tl, ot, mid = E.build(so, Lk, c["H"], c["feats"]); Fr = fold[tl]
            T = ((y[tl] == ot) & (y[tl] != Lk[tl])).astype(int)
            pk = E.train_pred(X, T, Fr != k, Fr == k, E.PARAMS, a.rounds)
            nk = R.flips(Lk, tl[Fr == k], ot[Fr == k], pk, K)[0]
            out[fold == k] = nk[fold == k]
            E.log(f"  fold {k}: stage1 {macro_f1(y[fold == k], L1[fold == k]):.4f} -> stage2 {macro_f1(y[fold == k], out[fold == k]):.4f}")
    E.log(f"RESULT stage2 {a.mode} {src}: F1 {macro_f1(y, fin):.4f} -> s1 {macro_f1(y, L1):.4f} -> s2 {macro_f1(y, out):.4f} | per fold "
          + " ".join(f"{macro_f1(y[fold == f], out[fold == f]):.4f}" for f in range(R.FOLDS)))
    np.save(os.path.join(R.HERE, "preds", f"{src}_s2{a.mode}_labo.npy"), out.astype(np.int8))


if __name__ == "__main__":
    main()
