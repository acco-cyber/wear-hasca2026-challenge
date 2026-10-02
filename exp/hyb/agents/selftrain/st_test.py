"""Cross-fitted self-training on the 4 test subjects: train S3/T on ALL OOF train rows (true labels) + test rows of the
other parts labelled with our best labels (weight w), predict test part k; rebuild the tab blend with the kernel's window logp.
python st_test.py --tag w03k2 --w 0.3 --K 2 [--filter none|c0.8|q0.5] [--wmode const|conf] [--out test_logp_b_st.npy]"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from st_cv import pseudo_mask
import lightgbm as lgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--w", type=float, default=0.3); ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--filter", default="none"); ap.add_argument("--wmode", default="const"); ap.add_argument("--group", default="chain")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pseed", type=int, default=0)
    ap.add_argument("--variants", default="S3,T"); ap.add_argument("--out", default="")
    ap.add_argument("--pl", default=os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv"))
    ap.add_argument("--ours", default=os.path.join(W, "subs", "sub_e44_vote9.csv"))
    a = ap.parse_args()
    out_dir = os.path.join(HERE, "runs_test", a.tag); os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out_dir, "args.json"), "w"))
    sm = load_meta(); y = sm["y"]
    bl = np.load(os.path.join(KEEP, "blend.npz")); sbj = bl["test_sbj"].astype(np.int64); nt = len(sbj)
    lab_pl = pd.read_csv(a.pl).sort_values("id").target_feature.to_numpy().astype(np.int64)
    s2, c2 = extra_links_test(nt); l2 = np.load(os.path.join(KEEP, "links_L2_test.npz"))
    conf = np.ones(nt)
    if a.filter != "none" or a.wmode == "conf":
        ours = pd.read_csv(a.ours).sort_values("id").target_feature.to_numpy().astype(np.int64)
        dd = dict(logp=np.load(os.path.join(KEEP, "test_logp_b.npy")).astype(np.float32),
                  emb=np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32), grp=sbj, sbj=sbj,
                  succ=l2["succ"].astype(np.int64), score=l2["score_qn"].astype(np.float32), sets={},
                  succ2=s2, score2=c2, xl_w=1.0, xl_b=-2.0, xlinks=[])
        lab, base, Q, P, g = recipe(dd, ours, np.ones(nt, bool))
        conf = Q.max(1); log(f"recipe re-run agrees with pseudo-label file {np.mean(lab == lab_pl):.4f}; conf>=0.8 {np.mean(conf >= 0.8):.3f}")
    grp = cross_groups([l2["succ"].astype(np.int64), s2], nt, sbj, cap=a.cap) if a.group == "chain" else np.arange(nt)
    te = np.arange(nt); part = np.zeros(nt, int); K = a.K if a.w > 0 else 1
    if K > 1:
        for s in np.unique(sbj):
            ii = np.flatnonzero(sbj == s); part[ii] = assign_parts(ii, grp, K, a.seed * 1000 + int(s))
    use = pseudo_mask(a.filter, conf, sbj, te)
    np.save(os.path.join(out_dir, "part.npy"), part)
    Xo, Xt = features("oof"), features("test"); params = dict(TAB_PARAMS); params["seed"] = a.pseed
    tr = np.arange(len(y)); OUT = {}
    for v in a.variants.split(","):
        outp = os.path.join(out_dir, f"{v}.npy")
        if os.path.exists(outp):
            OUT[v] = np.load(outp); log(f"{v}: exists"); continue
        Q = np.zeros((nt, N_CLS), np.float32)
        for k in range(K):
            ps = te[(part != k) & use] if a.w > 0 else te[:0]
            wt = np.concatenate([np.ones(len(tr)), a.w * (conf[ps] if a.wmode == "conf" else np.ones(len(ps)))])
            Xtr = np.concatenate([np.asarray(Xo[v]), np.asarray(Xt[v][ps])]); lab = np.concatenate([y, lab_pl[ps]])
            t0 = time.time(); bst = lgb.train(params, lgb.Dataset(Xtr, lab, weight=wt, params={"max_bin": 63}), ROUNDS[v]); del Xtr
            pk = te[part == k]; Q[pk] = np.log(np.clip(bst.predict(np.asarray(Xt[v][pk])), 1e-7, 1))
            log(f"{v} part {k}/{K}: train {len(tr)}+{len(ps)} pseudo, predict {len(pk)}, {time.time() - t0:.0f}s")
        np.save(outp, Q); OUT[v] = Q
        log(f"{v}: argmax agreement with pseudo-labels {np.mean(Q.argmax(1) == lab_pl):.4f}; with kernel tab_{v} {np.mean(Q.argmax(1) == bl['tab_' + v].argmax(1)):.4f}")
    if "S3" in OUT and "T" in OUT:
        LB = tab_blend(bl["test_logp"], OUT["S3"], OUT["T"]); kb = np.load(os.path.join(KEEP, "test_logp_b.npy"))
        out = a.out or os.path.join(out_dir, "test_logp_b.npy"); np.save(out, LB)
        log(f"wrote {out}; tab argmax agreement with kernel tab blend {np.mean(LB.argmax(1) == kb.argmax(1)):.4f}, with pseudo-labels {np.mean(LB.argmax(1) == lab_pl):.4f}")


if __name__ == "__main__":
    main()
