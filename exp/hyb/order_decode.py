"""Our mrf4 chain decoder (graph smoothing + chain Viterbi + count calibration + ICM) on top of the Hanbat graph
probabilities, along a given successor-link set. CV on the Hanbat OOF rows.
  python order_decode.py cv <oof_P.npy> --links true|l2|l0 [--variant mrf4] [--mixtrue p] [--per rec|sbj]
     true : true successors (oracle)        l2 : fold-honest OOF L2 links        l0 : OOF L0 links
     --mixtrue p : replace a fraction p of the link set by true successors (how much does link accuracy buy?)
  decode_links(P, succ, score, cand, lo, sbj, variant) is reused by the test script."""
import os, sys, argparse
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\decoder"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder import decode_subject, VARIANTS
from hanbat_stack import finish, macro_f1, KEEP, HB, TRAIN_SETS, N_CLS
K = 10


def make_struct(succ, score, cand=None, lo=None):
    """succ (n,) local successor or -1, score (n,) log-odds; optional extra candidates cand (n,m) / lo (n,m)"""
    n = len(succ); C = np.full((n, K), -1, np.int64); L = np.full((n, K), -50.0, np.float32)
    m = succ >= 0; C[:, 0] = succ; L[:, 0] = np.where(m, score, -50.0)
    if cand is not None:
        mm = min(cand.shape[1], K - 1); C[:, 1:1 + mm] = cand[:, :mm]; L[:, 1:1 + mm] = np.where(cand[:, :mm] >= 0, lo[:, :mm], -50.0)
        dup = C[:, 1:] == C[:, :1]; C[:, 1:][dup] = -1; L[:, 1:][dup] = -50.0
    return dict(cand=C, lo=L, succ0=succ.astype(np.int64), sc=np.where(m, score, -50.0).astype(np.float32),
                Lm=np.zeros((1, 1), np.float32), n=n)


def decode_links(P, succ_g, score_g, sbj, groups, variant="mrf4", cand_g=None, lo_g=None, cfg_over=None):
    """succ_g: global successor row index (or -1). Decodes each group (subject or recording) separately."""
    out = np.zeros(len(P), np.int64); cfg = dict(VARIANTS[variant]); cfg.update(cfg_over or {})
    for g in np.unique(groups):
        ii = np.flatnonzero(groups == g); pos = np.full(len(P), -1, np.int64); pos[ii] = np.arange(len(ii))
        su = succ_g[ii]; sl = np.where(su >= 0, pos[np.maximum(su, 0)], -1)
        # keep the link set one-to-one inside the group
        ok = sl >= 0; _, first = np.unique(np.where(ok, sl, -np.arange(1, len(ii) + 1)), return_index=True)
        keep = np.zeros(len(ii), bool); keep[first] = True; sl[~keep] = -1
        sc_l = score_g[ii]; seen = np.zeros(len(ii), bool)           # break every cycle at its weakest link
        for s0 in range(len(ii)):
            if seen[s0] or sl[s0] < 0:
                continue
            path, where, node = [], {}, s0
            while node >= 0 and not seen[node]:
                seen[node] = True; where[node] = len(path); path.append(node); node = int(sl[node])
            if node >= 0 and node in where:
                cyc = path[where[node]:]; sl[cyc[int(np.argmin([sc_l[x] for x in cyc]))]] = -1
        c = l = None
        if cand_g is not None:
            cg = cand_g[ii]; c = np.where(cg >= 0, pos[np.maximum(cg, 0)], -1); l = lo_g[ii]
        st = make_struct(sl, score_g[ii], c, l)
        out[ii] = decode_subject(P[ii], st, cfg, None)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("P"); ap.add_argument("--links", default="l2")
    ap.add_argument("--variant", default="mrf4"); ap.add_argument("--mixtrue", type=float, default=0.0); ap.add_argument("--per", default="sbj")
    ap.add_argument("--cands", action="store_true", help="add the L2 top candidates as extra graph edges")
    ap.add_argument("--score_true", type=float, default=4.0); ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--ns", type=float, default=None); ap.add_argument("--lo", type=int, default=None); ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--logp", action="store_true", help="P file holds log-probs"); ap.add_argument("--save", default="")
    a = ap.parse_args()
    P = np.load(a.P).astype(np.float64)
    if a.logp:
        P = np.exp(P - P.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    y, rec, sbj, fold = sm["y"], sm["rec"], sm["sbj"], sm["fold"]
    l2 = np.load(os.path.join(HB, "l2oof", "oof_L2.npz")); true_succ = l2["true_succ"].astype(np.int64)
    cand = lo = None
    if a.links == "true":
        succ, score = true_succ.copy(), np.full(len(y), a.score_true, np.float32)
    elif a.links == "l2":
        succ, score = l2["succ"].astype(np.int64), l2["score_qn"].astype(np.float32)
        if a.cands:
            cand, lo = l2["top_cand"].astype(np.int64), l2["top_lo"].astype(np.float32)
    elif a.links.endswith(".npz"):
        z = np.load(a.links); succ, score = z["succ"].astype(np.int64), z["score_qn"].astype(np.float32)
    else:
        l0 = np.load(os.path.join(KEEP, "links_L0.npz")); succ, score = l0["oof_succ"].astype(np.int64), l0["oof_score"].astype(np.float32)
    if a.mixtrue > 0:
        rng = np.random.default_rng(0); m = rng.random(len(y)) < a.mixtrue
        succ = np.where(m, true_succ, succ); score = np.where(m, np.maximum(score, a.score_true), score).astype(np.float32)
    lk = succ >= 0
    print(f"links {a.links} mixtrue {a.mixtrue}: linked {lk.mean():.3f} exact {np.mean(succ[lk] == true_succ[lk]):.3f} same-label {np.mean(y[succ[lk]] == y[lk]):.3f}")
    base = finish(P, dict(sbj=sbj, sets=TRAIN_SETS))
    over = {k: v for k, v in (("lam", a.lam), ("ns", a.ns), ("lo", a.lo), ("hi", a.hi)) if v is not None}
    out = decode_links(P, succ, score, sbj, sbj if a.per == "sbj" else rec, a.variant, cand, lo, over)
    f0, f1 = macro_f1(y, base), macro_f1(y, out)
    print(f"finish(P) {f0:.4f} | chain decode ({a.variant} {over}) {f1:.4f} ({f1 - f0:+.4f})  per fold "
          + " ".join(f"{macro_f1(y[fold == k], out[fold == k]):.4f}" for k in range(5)) + f"  null share true/pred {np.mean(y == 0):.3f}/{np.mean(out == 0):.3f}")
    if a.save:
        np.save(os.path.join(HB, f"{a.save}_labels.npy"), out)


if __name__ == "__main__":
    main()
