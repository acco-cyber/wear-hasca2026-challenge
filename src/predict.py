"""Test-time pipeline: per-window probs (work/<tag>/test.npy or a blend) -> per-subject structure -> decode -> submission.
python predict.py --tags lgbm_v1[,fusion_v1] [--weights 0.5,0.5] --mode graph|viterbi|graph_viterbi [--k 10 --alpha 0.7 --iters 8]
                  [--thr -4] [--p_stay 0.95] [--calib] [--lo 40 --hi 230] [--null_scale 1.0] [--out name.csv]
"""
import os, sys, time, argparse, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from chain import pair_features, assignment, cut, chains_from_succ
from decode import viterbi_chains, chain_vote, calibrate_counts, build_graph, graph_smooth
from simulate import load_prep, fit_scorer

DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep"); WORK = r"E:\Claude code\wear\work"; SUBS = r"E:\Claude code\wear\subs"
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
os.makedirs(SUBS, exist_ok=True); os.makedirs(WORK, exist_ok=True)

def get_scorer(path=os.path.join(WORK, "scorer.pkl"), sessions="sbj_1,sbj_3,sbj_7,sbj_12,sbj_16,sbj_19"):
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    meta, imu, vid = load_prep(); sc = fit_scorer(meta, imu, vid, sessions.split(","), np.random.RandomState(0))
    pickle.dump(sc, open(path, "wb")); return sc

def test_structure(cache=os.path.join(WORK, "test_structure.pkl")):
    """Per test subject: idx (global ids), cand, logodds, succ0, sc, Lm (assignment matrix)."""
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)   # match train quantisation
    vid = np.load(os.path.join(PREP, "test_vid_pca.npy"))
    limb = np.array([LIMBS.index(l) for l in tm.sensor_location])
    sc = get_scorer(); out = {}
    for s in sorted(tm.sbj_id.unique()):
        idx = np.where(tm.sbj_id.to_numpy() == s)[0]; n = len(idx); t0 = time.time()
        cand, F = pair_features(np.asarray(vid[idx], np.float32), xi[idx], limb[idx])
        lo = sc.logodds(F); succ0, scs, Lm = assignment(cand, lo, n)
        out[s] = dict(idx=idx, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm.astype(np.float16))
        print(f"sbj {s}: n={n} structure computed ({time.time()-t0:.0f}s); top logodds median {np.median(np.nanmax(np.where(np.isnan(lo), -99, lo), 1)):.2f}", flush=True)
    pickle.dump(out, open(cache, "wb")); return out

def load_probs(tags, weights):
    Ps = []
    for t in tags:
        P = np.load(os.path.join(WORK, t, "test.npy")).astype(np.float64); Ps.append(np.log(np.clip(P, 1e-6, 1)))
    L = sum(w * p for w, p in zip(weights, Ps)); P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)

def decode_subject(P, st, a):
    n = len(st["idx"])
    chains = chains_from_succ(cut(st["succ0"], st["sc"], st["Lm"].astype(np.float32), a.thr))
    if a.mode in ("graph", "graph_viterbi"):
        g = build_graph(st["cand"], st["lo"], n, k=a.k); P = graph_smooth(P, g, alpha=a.alpha, iters=a.iters)
    if a.calib:
        lab, b = calibrate_counts(P, chains if a.mode != "graph" else [[i] for i in range(n)], lo=a.lo, hi=a.hi, p_stay=a.p_stay)
        return lab, chains, P
    if a.mode == "graph": return P.argmax(1), chains, P
    return viterbi_chains(P, chains, p_stay=a.p_stay), chains, P

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="lgbm_v1"); ap.add_argument("--weights", default=None)
    ap.add_argument("--mode", default="graph_viterbi"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--thr", type=float, default=-4.0); ap.add_argument("--p_stay", type=float, default=0.95)
    ap.add_argument("--calib", action="store_true"); ap.add_argument("--lo", type=int, default=40); ap.add_argument("--hi", type=int, default=230)
    ap.add_argument("--null_scale", type=float, default=1.0); ap.add_argument("--out", default=None)
    ap.add_argument("--votes", default=None, help="comma list of csv_path:weight added to log-probs of the voted class")
    ap.add_argument("--probs", default=None, help="use this (12234,19) .npy instead of --tags")
    a = ap.parse_args()
    tags = a.tags.split(","); weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0 / len(tags)] * len(tags)
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    P_all = load_probs(tags, weights) if not a.probs else np.load(a.probs).astype(np.float64)
    if a.votes:
        L = np.log(np.clip(P_all, 1e-9, 1))
        for spec in a.votes.split(","):
            path, w = spec.rsplit(":", 1)
            v = pd.read_csv(path).sort_values("id").iloc[:, 1].to_numpy().astype(int)
            L[np.arange(len(v)), v] += float(w)
        P_all = np.exp(L - L.max(1, keepdims=True)); P_all /= P_all.sum(1, keepdims=True)
        print("votes applied:", a.votes)
    P_all[:, 0] *= a.null_scale; P_all /= P_all.sum(1, keepdims=True)
    raw = P_all.argmax(1); lab = raw.copy(); struct = test_structure()
    for s, st in struct.items():
        idx = st["idx"]; l, chains, Ps = decode_subject(P_all[idx], st, a); lab[idx] = l
        lens = np.array([len(c) for c in chains])
        print(f"sbj {s}: n={len(idx)} chains={len(chains)} max={lens.max()} | null raw {np.mean(raw[idx]==0):.3f} -> {np.mean(l==0):.3f} | "
              f"changed {np.mean(l!=raw[idx]):.3f} | classes {np.unique(l).size} | counts {np.bincount(l, minlength=19).tolist()}", flush=True)
    name = a.out or f"sub_{'+'.join(tags)}_{a.mode}_k{a.k}a{a.alpha}_thr{a.thr}_ps{a.p_stay}{'_cal' if a.calib else ''}_ns{a.null_scale}.csv"
    pd.DataFrame({"id": tm.id, "target_feature": lab}).to_csv(os.path.join(SUBS, name), index=False)
    print("overall null", round(float((lab == 0).mean()), 3), "| wrote", os.path.join(SUBS, name))

if __name__ == "__main__":
    main()
