"""Cross-fitted self-training of the tabular experts, simulated on OOF.
For held-out fold f: train rows = folds != f (true labels, weight 1) + fold-f rows of the OTHER parts with recipe pseudo-labels
(weight w, optional confidence filter); predict fold-f rows of part k.  Outputs runs/<tag>/<variant>_f<f>.npy (logp of fold-f rows).
python st_cv.py --tag w03k2 --w 0.3 --K 2 [--filter none|c0.8|q0.5] [--wmode const|conf] [--group chain|random] [--folds 0,1]"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import lightgbm as lgb


def pseudo_mask(filt, conf, sbj, rows):
    m = np.ones(len(rows), bool)
    if filt.startswith("c"):
        m = conf[rows] >= float(filt[1:])
    elif filt.startswith("q"):
        q = float(filt[1:]); s = sbj[rows]
        for u in np.unique(s):
            ii = s == u; m[ii] = conf[rows][ii] >= np.quantile(conf[rows][ii], q)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--w", type=float, default=0.3); ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--filter", default="none"); ap.add_argument("--wmode", default="const"); ap.add_argument("--group", default="chain")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pseed", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--variants", default="S3,T")
    ap.add_argument("--pl", default="lab", help="field of pl_oof.npz used as pseudo-label")
    a = ap.parse_args()
    out_dir = os.path.join(HERE, "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out_dir, "args.json"), "w"))
    sm = load_meta(); y, fold, sbj = sm["y"], sm["fold"], sm["sbj"]; n = len(y)
    pl = np.load(os.path.join(HERE, "pl_oof.npz")); lab_pl, conf = pl[a.pl].astype(np.int64), pl["conf"]
    X = features("oof")
    if a.group == "chain":
        l0 = np.load(os.path.join(KEEP, "links_L0.npz")); s2, _ = extra_links_oof(n)
        grp = cross_groups([l0["oof_succ"].astype(np.int64), s2], n, sbj, cap=a.cap)
    else:
        grp = np.arange(n)
    params = dict(TAB_PARAMS); params["seed"] = a.pseed
    for f in [int(x) for x in a.folds.split(",")]:
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
        K = a.K if a.w > 0 else 1
        part = np.zeros(len(te), int)
        if K > 1:
            for s in np.unique(sbj[te]):                       # balance parts within every subject
                ii = np.flatnonzero(sbj[te] == s); part[ii] = assign_parts(te[ii], grp, K, a.seed * 1000 + int(s))
        use = pseudo_mask(a.filter, conf, sbj, te)
        np.save(os.path.join(out_dir, f"part_f{f}.npy"), part)
        for v in a.variants.split(","):
            outp = os.path.join(out_dir, f"{v}_f{f}.npy")
            if os.path.exists(outp):
                log(f"fold {f} {v}: exists, skipped"); continue
            Xv = X[v]; Q = np.zeros((len(te), N_CLS), np.float32)
            for k in range(K):
                ps = te[(part != k) & use] if a.w > 0 else te[:0]
                idx = np.concatenate([tr, ps]); lab = np.concatenate([y[tr], lab_pl[ps]])
                wt = np.concatenate([np.ones(len(tr)), a.w * (conf[ps] if a.wmode == "conf" else np.ones(len(ps)))])
                t0 = time.time()
                ds = lgb.Dataset(np.asarray(Xv[idx]), lab, weight=wt, params={"max_bin": 63})
                bst = lgb.train(params, ds, ROUNDS[v])
                pk = te[part == k]
                Q[part == k] = np.log(np.clip(bst.predict(np.asarray(Xv[pk])), 1e-7, 1))
                log(f"fold {f} {v} part {k}/{K}: train {len(tr)}+{len(ps)} pseudo, predict {len(pk)}, {time.time() - t0:.0f}s")
            np.save(outp, Q)
            log(f"fold {f} {v}: F1 {macro_f1(y[te], Q.argmax(1)):.4f}")


if __name__ == "__main__":
    main()
