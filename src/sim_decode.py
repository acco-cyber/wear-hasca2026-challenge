"""Fast decoder experiments on cached simulation structure (no scorer refit): loads OOF probs, rebuilds per-session
structure once (cached to work/sim_struct.pkl), then evaluates decoder variants.
python sim_decode.py --oof work/lgbm_v1/oof.npy [--oof2 work/fusion_v1/oof.npy --w2 0.5]
"""
import os, sys, time, argparse, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from chain import make_training_pairs, assignment, cut, chains_from_succ
from decode import viterbi_chains, calibrate_counts, build_graph, graph_smooth
from simulate import load_prep, session_slices, fit_scorer

WORK = r"E:\Claude code\wear\work"

def get_struct(sessions, cache=os.path.join(WORK, "sim_struct.pkl")):
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    meta, imu, vid = load_prep(); rng = np.random.RandomState(0)
    sc = pickle.load(open(os.path.join(WORK, "scorer.pkl"), "rb")); out = {}
    for s in sessions:
        a, b = session_slices(meta)[s]; n = b - a
        F, L, cand, limb, Xi = make_training_pairs(np.asarray(vid[a:b], np.float32), np.asarray(imu[a:b], np.float32), rng)
        lo = sc.logodds(F); succ0, scs, Lm = assignment(cand, lo, n)
        out[s] = dict(a=a, b=b, n=n, y=meta.y.to_numpy()[a:b], limb=limb, cand=cand, lo=lo, succ0=succ0, sc=scs, Lm=Lm)
        print("struct", s, flush=True)
    pickle.dump(out, open(cache, "wb")); return out

def ipf_targets(P, n, act_sec=100.0, iters=60, step=0.3):
    """Additive biases so that argmax counts per activity approach act_sec each (soft, via bisection-like updates)."""
    b = np.zeros(19); logP = np.log(np.clip(P, 1e-6, 1))
    for _ in range(iters):
        lab = (logP + b).argmax(1); cnt = np.bincount(lab, minlength=19)[1:]
        b[1:] += step * np.sign(act_sec - cnt) * (np.abs(act_sec - cnt) > 15)
    return (logP + b).argmax(1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--oof", required=True); ap.add_argument("--oof2", default=None); ap.add_argument("--w2", type=float, default=0.5)
    ap.add_argument("--sessions", default="sbj_0,sbj_5,sbj_10,sbj_14_2,sbj_20,sbj_21")
    a = ap.parse_args(); sessions = a.sessions.split(","); S = get_struct(sessions)
    oof = np.load(a.oof)
    if a.oof2:
        o2 = np.load(a.oof2); oof = np.exp((1 - a.w2) * np.log(np.clip(oof, 1e-6, 1)) + a.w2 * np.log(np.clip(o2, 1e-6, 1)))
        oof /= np.nansum(oof, 2, keepdims=True)
    rows = []
    for s in sessions:
        st = S[s]; n = st["n"]; y = st["y"]
        P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]]; ok = ~np.isnan(P[:, 0]); P = np.where(ok[:, None], P, 1.0 / 19)
        res = dict(session=s, null=round(float((y == 0).mean()), 3), raw=f1_score(y, P.argmax(1), average="macro"))
        res["ipf100"] = f1_score(y, ipf_targets(P, n), average="macro")
        g = build_graph(st["cand"], st["lo"], n, k=10); Pg = graph_smooth(P, g, alpha=0.5, iters=5)
        res["g"] = f1_score(y, Pg.argmax(1), average="macro"); res["g_ipf100"] = f1_score(y, ipf_targets(Pg, n), average="macro")
        single = [[i] for i in range(n)]
        for lo_, hi_ in ((40, 230), (60, 160), (70, 130)):
            lab, _ = calibrate_counts(Pg, single, lo=lo_, hi=hi_); res[f"g_cal{lo_}_{hi_}"] = f1_score(y, lab, average="macro")
        chains = chains_from_succ(cut(st["succ0"], st["sc"], st["Lm"], -6.0))
        for ps in (0.8, 0.9):
            lab, _ = calibrate_counts(Pg, chains, lo=60, hi=160, p_stay=ps); res[f"g_chain_cal_ps{ps}"] = f1_score(y, lab, average="macro")
        # stronger smoothing
        Pg2 = graph_smooth(P, build_graph(st["cand"], st["lo"], n, k=15), alpha=0.7, iters=10)
        lab, _ = calibrate_counts(Pg2, single, lo=60, hi=160); res["g15_0.7_cal60_160"] = f1_score(y, lab, average="macro")
        # oracle order references
        full = [list(range(n))]
        lab, _ = calibrate_counts(P, full, lo=60, hi=160, p_stay=0.95); res["oracle_cal60_160"] = f1_score(y, lab, average="macro")
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}); print(rows[-1], flush=True)
    df = pd.DataFrame(rows); pd.set_option("display.width", 250); print(df.to_string(index=False)); print("MEAN:", df.mean(numeric_only=True).round(4).to_dict())

if __name__ == "__main__":
    main()
