"""Per-subject decoder: per-window class probabilities + reconstructed link structure -> labels.

Pipeline per subject (variant 'v1' defaults):
  1. neighbour graph = top-k link-scorer successors/predecessors (weights sigmoid(log-odds))
     [+ optional within-subject video kNN edges on mean-pooled PCA-160 frame descriptors]
  2. log-probability graph smoothing (alpha, iters)
  3. chains from linear-assignment successors cut at log-odds thr
  4. chain Viterbi (numba) with per-edge switch cost scaled by link confidence (2*sigmoid(beta*log-odds)),
     class biases calibrated so every activity class gets between lo and hi windows (null takes the rest)
  [5. optional Potts ICM refinement on the symmetric graph inside the calibration loop]

CLI (test):
  python decoder.py --probs P.npy [--variant v1] [--struct work\\test_structure.pkl] [--out_name sub_decoder_v1.csv]
  python decoder.py --blend "lgbm_v1:0.8,fusion_v1:0.2" --variant v1      (log-space blend of work\\<tag>\\test.npy)
"""
import os, sys, argparse, pickle, json
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import graph_matrix, smooth, NC, WORK, PREP
from nbvit import pack, calib, logT_matrix, viterbi
from mrf import calib_mrf
from mrf2 import calib_mrf2
from vidknn import mp_desc, knn_edges
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import cut, chains_from_succ

DATA = r"E:\Claude code\wear\data"; SUBS = r"E:\Claude code\wear\subs"

VARIANTS = {
    # baseline decoder of e1 (for reference)
    "base": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=60, hi=160, ps=0.8, beta=None, lam=0.0),
    # tighter count bounds
    "c85": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=85, hi=160, ps=0.8, beta=None, lam=0.0),
    # lead's decoder (e2): null prob x0.5 before smoothing, bounds 80..250
    "lead": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=80, hi=250, ps=0.8, beta=None, lam=0.0, ns=0.5),
    # chain-Viterbi init + Potts ICM refinement on the symmetric link graph inside the count-calibration loop
    "mrf3": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=85, hi=160, ps=0.8, beta=None, lam=3.0, icm_iters=10),
    # best (sweep G): lead's null scaling + bounds, Potts lambda 4, p_stay 0.7
    "mrf4": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=80, hi=250, ps=0.7, beta=None, lam=4.0, icm_iters=10, ns=0.5),
    # mrf4 with flattened MRF link weights sigmoid(lo)^0.5 (sweep K: +0.009 extra, -0.004 eval averaged over 3 seeds)
    "mrf4p05": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=80, hi=250, ps=0.7, beta=None, lam=4.0, icm_iters=10, ns=0.5, power=0.5),
    # majority vote of 6 mrf4 decodes: link-weight power {1, 0.5} x ICM seed {0,1,2} (sweep L)
    "mix6": dict(k=10, alpha=0.5, iters=5, kv=0, wv=0.0, thr=-6.0, lo=80, hi=250, ps=0.7, beta=None, lam=4.0, icm_iters=10, ns=0.5,
                 ens=[(1.0, 0), (1.0, 1), (1.0, 2), (0.5, 0), (0.5, 1), (0.5, 2)]),
}
# 2026-09-25 LB knobs on top of mrf4
VARIANTS["mrf4_lo90"] = dict(VARIANTS["mrf4"], lo=90)
VARIANTS["mrf4_ns06"] = dict(VARIANTS["mrf4"], ns=0.6)

def edge_weights(packed, sc, beta, center=0.0):
    order, starts, lens = packed
    prev = np.r_[-1, order[:-1]]; is_start = np.zeros(len(order), bool); is_start[starts] = True
    e = np.where(is_start, 0.0, sc[np.maximum(prev, 0)])
    return np.where(is_start, 1.0, 2 / (1 + np.exp(-beta * (e - center))))

def decode_subject(P, st, cfg, vid=None, return_info=False):
    """P (n,19) probabilities of one subject's windows (rows aligned with st's windows); st: dict with cand, lo, succ0,
    sc, Lm; vid: optional (n,15,160) PCA frame features for video kNN edges. Returns labels (n,)."""
    n = len(P); c = cfg
    if c.get("ns", 1.0) != 1.0:
        P = P.copy(); P[:, 0] *= c["ns"]; P = P / P.sum(1, keepdims=True)
    extra = None
    if c.get("wv", 0) > 0 and vid is not None:
        D = mp_desc(vid); r, cc, w = knn_edges(D, c["kv"]); w = np.full(len(r), c["wv"])
        extra = (np.r_[r, cc], np.r_[cc, r], np.r_[w, w])
    W = graph_matrix(st["cand"], st["lo"], n, k=c["k"], extra=extra)
    logP = np.log(np.clip(smooth(P, W, c["alpha"], c["iters"]), 1e-6, 1)) if c["alpha"] > 0 else np.log(np.clip(P, 1e-6, 1))
    chains = chains_from_succ(cut(st["succ0"], st["sc"], np.asarray(st["Lm"], np.float32), c["thr"])); pk = pack(chains)
    ew = edge_weights(pk, np.asarray(st["sc"], np.float64), c["beta"], c.get("center", 0.0)) if c.get("beta") else None
    Wm = None
    if c.get("ens"):                       # majority vote over MRF decodes with different weight powers / ICM seeds
        W0 = graph_matrix(st["cand"], st["lo"], n, k=c["k"]); V = np.zeros((n, NC)); Ws = {}
        for pw, sd in c["ens"]:
            if pw not in Ws:
                Wp = W0.power(pw) if pw != 1.0 else W0; Wp = Wp + Wp.T; d = np.asarray(Wp.sum(1)).ravel()
                Ws[pw] = (sp.diags(1 / np.where(d > 0, d, 1)) @ Wp).tocsr()
            l_, b = calib_mrf2(logP, pk, Ws[pw], c["lo"], c["hi"], c["ps"], c["lam"], None, c.get("icm_iters", 10), seed=sd)
            V[np.arange(n), l_] += 1
        lab = V.argmax(1); Wm = Ws[c["ens"][0][0]]
    elif c.get("lam", 0) > 0:
        Wm = graph_matrix(st["cand"], st["lo"], n, k=c["k"])
        if c.get("power", 1.0) != 1.0: Wm = Wm.power(c["power"])          # flatten link-confidence weights
        Wm = Wm + Wm.T
        d = np.asarray(Wm.sum(1)).ravel(); Wm = (sp.diags(1 / np.where(d > 0, d, 1)) @ Wm).tocsr()
        if c.get("power", 1.0) == 1.0 and c.get("vote", 1) == 1 and c.get("seed", 0) == 0:
            lab, b = calib_mrf(logP, pk, Wm, c["lo"], c["hi"], c["ps"], c["lam"], c.get("icm_iters", 10))
        else:
            labs = []
            for v in range(c.get("vote", 1)):
                l_, b = calib_mrf2(logP, pk, Wm, c["lo"], c["hi"], c["ps"], c["lam"], None, c.get("icm_iters", 10), seed=c.get("seed", 0) + v)
                labs.append(l_)
            V = np.zeros((n, NC))
            for l_ in labs: V[np.arange(n), l_] += 1
            lab = V.argmax(1) if len(labs) > 1 else labs[0]
    else:
        lab, b = calib(logP, pk, c["lo"], c["hi"], p_stay=c["ps"], ew=ew)
    if c.get("post"):                      # optional post-processing hook (family disambiguation etc.)
        lab = c["post"](lab, dict(logP=logP, b=b, W=Wm, P=P, vid=vid, st=st, cfg=c))
    if return_info: return lab, dict(bias=b, chains=len(chains), logP=logP, W=Wm)
    return lab

def load_blend(spec):
    Ls = []
    for part in spec.split(","):
        tag, w = part.split(":"); P = np.load(os.path.join(WORK, tag, "test.npy")).astype(np.float64)
        Ls.append(float(w) * np.log(np.clip(P, 1e-6, 1)))
    L = sum(Ls); P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)

def sim_eval(variants):
    """Evaluate variants through decode_subject on the 6 eval + 6 extra sim sessions (0.8/0.2 OOF blend)."""
    from common import load_oof, load_structs, sess_P, mf1, EVAL, EXTRA
    from vidknn import train_vid
    oof = load_oof(0.2); S = load_structs(("eval", "extra")); rows = []
    for s in [x for x in EVAL + EXTRA if x in S]:
        st = S[s]; P = sess_P(oof, st); r = dict(session=s)
        for v in variants:
            cfg = VARIANTS[v]; vid = np.asarray(train_vid()[st["a"]:st["b"]], np.float32) if cfg.get("wv", 0) > 0 else None
            r[v] = mf1(st["y"], decode_subject(P, st, cfg, vid))
        rows.append(r); print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
    df = pd.DataFrame(rows).set_index("session"); ex = [s for s in df.index if s not in EVAL]
    df.loc["EVAL_MEAN"] = df.loc[EVAL].mean(); df.loc["EXTRA_MEAN"] = df.loc[ex].mean()
    pd.set_option("display.width", 250); print(df.round(4).to_string()); return df

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--sim":
        df = sim_eval(sys.argv[2].split(",")); df.to_csv(os.path.join(HERE, f"sim_{sys.argv[2].replace(',', '+')}.csv")); return
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=None, help="(12234,19) .npy of per-window test probabilities (row = test id)")
    ap.add_argument("--blend", default=None, help='e.g. "lgbm_v1:0.8,fusion_v1:0.2" (log-space blend of work/<tag>/test.npy)')
    ap.add_argument("--struct", default=os.path.join(WORK, "test_structure.pkl"))
    ap.add_argument("--variant", default="v1"); ap.add_argument("--out_name", default=None)
    ap.add_argument("--save_probs", default=None, help="where to save the probabilities used (.npy)")
    a = ap.parse_args(); cfg = VARIANTS[a.variant]
    if a.probs: P_all = np.load(a.probs).astype(np.float64); P_all = P_all / P_all.sum(1, keepdims=True)
    else: P_all = load_blend(a.blend)
    assert P_all.shape == (12234, NC), P_all.shape
    if a.save_probs: np.save(a.save_probs, P_all.astype(np.float32))
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv")); assert (tm.id.to_numpy() == np.arange(len(tm))).all()
    struct = pickle.load(open(a.struct, "rb"))
    vid = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r") if cfg.get("wv", 0) > 0 else None
    lab = P_all.argmax(1).copy()
    print("variant", a.variant, json.dumps(cfg))
    for s in sorted(struct):
        st = struct[s]; idx = st["idx"]
        l, info = decode_subject(P_all[idx], st, cfg, None if vid is None else np.asarray(vid[idx], np.float32), return_info=True)
        lab[idx] = l; cnt = np.bincount(l, minlength=NC)
        print(f"sbj {s}: n={len(idx)} chains={info['chains']} null raw {np.mean(P_all[idx].argmax(1) == 0):.3f} -> decoded {cnt[0] / len(idx):.3f} "
              f"(1-1800/n = {1 - 1800 / len(idx):.3f}) | activity counts {cnt[1:].tolist()}", flush=True)
    name = a.out_name or f"sub_decoder_{a.variant}.csv"; out = os.path.join(SUBS, name)
    pd.DataFrame({"id": tm.id.to_numpy(), "target_feature": lab}).to_csv(out, index=False)
    print("overall null", round(float((lab == 0).mean()), 3), "| classes present", np.unique(lab).size, "| wrote", out)

if __name__ == "__main__":
    main()
