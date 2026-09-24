"""De-risking experiment: simulate the test construction on held-out TRAIN sessions (known order),
reconstruct chains, and (optionally) decode OOF class probabilities along chains.

python simulate.py                              -> chain quality, threshold sweep
python simulate.py --oof work/lgbm_v1/oof.npy   -> also end-to-end macro-F1 raw vs chain-decoded
"""
import os, sys, json, time, argparse, pickle
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from chain import make_training_pairs, Scorer, assignment, cut, chains_from_succ, edge_metrics, FEAT_NAMES
from decode import viterbi_chains, chain_vote, calibrate_counts, build_graph, graph_smooth

DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep"); WORK = r"E:\Claude code\wear\work"
os.makedirs(WORK, exist_ok=True)

def load_prep():
    meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    imu = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r")
    vid = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
    return meta, imu, vid

def session_slices(meta):
    return {s: (g.index.min(), g.index.max() + 1) for s, g in meta.groupby("session", sort=False)}

def fit_scorer(meta, imu, vid, sessions, rng, neg_per_pos=30):
    Fs, Ls = [], []; sl = session_slices(meta)
    for s in sessions:
        a, b = sl[s]
        F, L, cand, limb, Xi = make_training_pairs(np.asarray(vid[a:b], np.float32), np.asarray(imu[a:b], np.float32), rng)
        F2 = F.reshape(-1, F.shape[-1]); L2 = L.reshape(-1); ok = ~np.isnan(F2[:, 0])
        pos = np.where(ok & (L2 == 1))[0]; neg = np.where(ok & (L2 == 0))[0]
        keep = rng.choice(neg, min(len(neg), neg_per_pos * len(pos)), replace=False)
        sel = np.concatenate([pos, keep]); Fs.append(F2[sel]); Ls.append(L2[sel])
        print(f"  scorer pairs from {s}: pos {len(pos)} / {len(L)-1} windows (true successor among candidates: {len(pos)/(len(L)-1):.3f})", flush=True)
    sc = Scorer().fit(np.concatenate(Fs), np.concatenate(Ls))
    imp = sorted(zip(FEAT_NAMES, sc.m.feature_importances_), key=lambda x: -x[1]); print("  scorer importances:", imp, flush=True)
    return sc

def simulate_session(meta, imu, vid, s, scorer, rng):
    a, b = session_slices(meta)[s]; n = b - a; y = meta.y.to_numpy()[a:b]
    F, L, cand, limb, Xi = make_training_pairs(np.asarray(vid[a:b], np.float32), np.asarray(imu[a:b], np.float32), rng)
    lo = scorer.logodds(F); succ0, sc, Lm = assignment(cand, lo, n)
    return dict(n=n, y=y, limb=limb, succ0=succ0, sc=sc, Lm=Lm, cand=cand, lo=lo, ceiling=float(L.sum() / (n - 1)))

def eval_threshold(r, thr):
    succ = cut(r["succ0"], r["sc"], r["Lm"], thr); n = r["n"]; y = r["y"]
    em = edge_metrics(succ, n); chains = chains_from_succ(succ); e = succ >= 0
    lens = np.array([len(c) for c in chains])
    em.update(same_label=float((y[e] == y[succ[e]]).mean()) if e.any() else 0.0, chains=len(chains), max_chain=int(lens.max()),
              frac_ge50=float(lens[lens >= 50].sum() / n), frac_ge20=float(lens[lens >= 20].sum() / n))
    return em, chains

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--oof", default=None)
    ap.add_argument("--thr", default="-6,-4,-3,-2,-1,0"); ap.add_argument("--fit_sessions", default="sbj_1,sbj_3,sbj_7,sbj_12,sbj_16,sbj_19")
    ap.add_argument("--eval_sessions", default="sbj_0,sbj_5,sbj_10,sbj_14_2,sbj_20,sbj_21")
    ap.add_argument("--p_stay", type=float, default=0.95); ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--save_scorer", default=os.path.join(WORK, "scorer.pkl"))
    a = ap.parse_args(); rng = np.random.RandomState(0); t0 = time.time()
    meta, imu, vid = load_prep()
    print("fitting scorer", flush=True)
    sc = fit_scorer(meta, imu, vid, a.fit_sessions.split(","), rng)
    pickle.dump(sc, open(a.save_scorer, "wb"))
    oof = np.load(a.oof) if a.oof else None
    thrs = [float(x) for x in a.thr.split(",")]; rows = []
    for s in a.eval_sessions.split(","):
        r = simulate_session(meta, imu, vid, s, sc, rng)
        print(f"{s}: n={r['n']} candidate ceiling {r['ceiling']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        for thr in thrs:
            em, chains = eval_threshold(r, thr)
            line = dict(session=s, thr=thr, edges=em["n_edges"], prec=round(em["precision"], 3), rec=round(em["recall"], 3),
                        same_lab=round(em["same_label"], 3), chains=em["chains"], max_chain=em["max_chain"], f_ge20=round(em["frac_ge20"], 3), f_ge50=round(em["frac_ge50"], 3))
            if oof is not None:
                aa, bb = session_slices(meta)[s]; P = oof[aa:bb][np.arange(r["n"]), r["limb"]]
                ok = ~np.isnan(P[:, 0]); P = np.where(ok[:, None], P, 1.0 / 19); y = r["y"]
                f_raw = f1_score(y, P.argmax(1), average="macro")
                f_vit = f1_score(y, viterbi_chains(P, chains, p_stay=a.p_stay, temp=a.temp), average="macro")
                lab_cal, _ = calibrate_counts(P, chains, p_stay=a.p_stay, temp=a.temp); f_cal = f1_score(y, lab_cal, average="macro")
                line.update(f1_raw=round(f_raw, 4), f1_vit=round(f_vit, 4), f1_cal=round(f_cal, 4))
                if thr == thrs[0]:
                    f_orc = f1_score(y, viterbi_chains(P, [list(range(r["n"]))], p_stay=a.p_stay, temp=a.temp), average="macro")
                    lab_oc, _ = calibrate_counts(P, [list(range(r["n"]))], p_stay=a.p_stay, temp=a.temp)
                    line.update(f1_oracle=round(f_orc, 4), f1_oracle_cal=round(f1_score(y, lab_oc, average="macro"), 4), null_true=round(float((y == 0).mean()), 3))
                    # graph smoothing variants (independent of threshold)
                    for kk, al, it in ((10, 0.5, 5), (10, 0.7, 8), (20, 0.7, 8)):
                        g = build_graph(r["cand"], r["lo"], r["n"], k=kk)
                        Pg = graph_smooth(P, g, alpha=al, iters=it)
                        f_g = f1_score(y, Pg.argmax(1), average="macro")
                        f_gv = f1_score(y, viterbi_chains(Pg, chains, p_stay=a.p_stay, temp=a.temp), average="macro")
                        lab_gc, _ = calibrate_counts(Pg, chains, p_stay=a.p_stay, temp=a.temp)
                        line[f"g{kk}_{al}"] = round(f_g, 4); line[f"g{kk}_{al}_vit"] = round(f_gv, 4); line[f"g{kk}_{al}_cal"] = round(f1_score(y, lab_gc, average="macro"), 4)
            rows.append(line)
        print(pd.DataFrame([x for x in rows if x["session"] == s]).to_string(index=False), flush=True)
    df = pd.DataFrame(rows); print("=== mean over sessions by threshold ==="); print(df.groupby("thr").mean(numeric_only=True).round(4).to_string())
    df.to_csv(os.path.join(WORK, "simulate_last.csv"), index=False)

if __name__ == "__main__":
    main()
