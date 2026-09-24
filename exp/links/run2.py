"""Links v2 pipeline on the simulation: encoder scores -> pair features v2 -> LightGBM scorer -> assignment -> decode.
python run2.py --enc enc_try.pt --fit sbj_1,sbj_3,... --tag v2 [--draws 2]
"""
import argparse, time
from lk import *
from feat2 import pair_features2, FEAT2
import enc as E
import torch, lightgbm as lgb

def enc_scores(models, V, Xi, limb):
    vp, ip = E.session_inputs(V, Xi=Xi, limb=limb)
    return np.mean([E.score_matrix(m, vp, ip) for m in models], 0)

def load_models(names):
    out = []
    for nm in names:
        sd = torch.load(os.path.join(EXP, nm)); din = sd["f.net.1.weight"].shape[1]; m = E.BiEnc(din); m.load_state_dict(sd); m.eval(); out.append(m)
    return out

def session_pairs(models, V, X4, limb, **kw):
    n = len(V); Xi = X4[np.arange(n), limb]
    Snn = enc_scores(models, V, Xi, limb)
    cand, F = pair_features2(V, Xi, limb, Snn, **kw)
    lab = (cand == (np.arange(n)[:, None] + 1)).astype(np.int64); lab[-1] = 0
    return cand, F, lab

def fit_scorer(Fs, Ls, rounds=400, seed=0):
    F = np.concatenate([f.reshape(-1, f.shape[-1]) for f in Fs]); y = np.concatenate([l.reshape(-1) for l in Ls])
    ok = ~np.isnan(F[:, 0]); F, y = F[ok], y[ok]
    rng = np.random.RandomState(seed); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    neg = rng.choice(neg, min(len(neg), 40 * len(pos)), replace=False); sel = np.concatenate([pos, neg])
    m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=3)
    m.fit(F[sel], y[sel]); print("  scorer pos", len(pos), "neg", len(neg), "imp", sorted(zip(FEAT2, m.feature_importances_), key=lambda x: -x[1])[:12], flush=True)
    return m

def logodds(m, F):
    sh = F.shape[:-1]; Z = F.reshape(-1, F.shape[-1]); out = np.full(len(Z), -50.0, np.float32); ok = ~np.isnan(Z[:, 0])
    p = np.clip(m.predict_proba(Z[ok])[:, 1], 1e-6, 1 - 1e-6); out[ok] = np.log(p / (1 - p)); return out.reshape(sh)

def valid_limbs(X4, rng):
    return E.valid_limb_choice(X4, rng)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--enc", required=True); ap.add_argument("--fit", default=",".join(OLD_FIT))
    ap.add_argument("--tag", default="v2"); ap.add_argument("--draws", type=int, default=1); ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args(); t0 = time.time()
    models = load_models(a.enc.split(",")); meta, imu, vid = load_prep(); sl = session_slices(meta); rng = np.random.RandomState(1)
    Fs, Ls = [], []
    for s in a.fit.split(","):
        a_, b_ = sl[s]; V = np.asarray(vid[a_:b_], np.float32); X4 = np.asarray(imu[a_:b_], np.float32)
        for d in range(a.draws):
            limb = valid_limbs(X4, rng); cand, F, lab = session_pairs(models, V, X4, limb)
            Fs.append(F); Ls.append(lab)
            print(f"fit {s} draw {d}: cand recall {lab[:-1].any(1).mean():.3f} M={cand.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    scorer = fit_scorer(Fs, Ls, rounds=a.rounds); pickle.dump(scorer, open(os.path.join(EXP, f"scorer_{a.tag}.pkl"), "wb")); del Fs, Ls
    S0 = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); oof = blend_oof(0.2); out = {}; rows = []
    for s in EVAL:
        st0 = S0[s]; a_, b_ = st0["a"], st0["b"]; n = st0["n"]; y = st0["y"]; limb = st0["limb"]
        V = np.asarray(vid[a_:b_], np.float32); X4 = np.asarray(imu[a_:b_], np.float32)
        cand, F, lab = session_pairs(models, V, X4, limb)
        lo = logodds(scorer, F); succ0, sc, Lm = assignment_from(cand, lo, n)
        st = dict(a=a_, b=b_, n=n, y=y, limb=limb, cand=cand, lo=lo, succ0=succ0, sc=sc, Lm=Lm); out[s] = st
        P = sim_P(oof, st)
        r = dict(session=s, f1_old=f1(y, decode(P, cand, lo, succ0, sc, Lm, **DEC_OLD)), f1_new=f1(y, decode(P, cand, lo, succ0, sc, Lm, **DEC_NEW)),
                 **link_metrics(cut(succ0, sc, Lm, -6.0), y, cand), prec_all=float((succ0[:-1] == np.arange(1, n)).mean()))
        rows.append(r); print(r, f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); print(df.round(4).to_string()); print("MEAN", df.mean(numeric_only=True).round(4).to_dict())
    pickle.dump(out, open(os.path.join(EXP, f"sim_struct_{a.tag}.pkl"), "wb"))
