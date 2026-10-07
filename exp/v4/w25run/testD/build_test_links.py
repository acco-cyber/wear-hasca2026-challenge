"""Task D, steps (3)-(6): re-score the L3 test link candidates of a fit with chain evidence from the real 2025 data.
Training = build_links.py --chain lgb --stage2 features on all 22 training subjects (cached nested sim chains); stage-1 and
stage-2 LightGBM (init_score = L3 log-odds, same params / subsampling / DROP) trained nested (OOF links, fold models for the
self-check) and on all subjects (test). Test tables use per-limb chain node sets (tdlib.subject_table).
  python build_test_links.py --fit K7 --mode spec [--selfcheck 6,18,20]
  mode spec: anchor 1.0 in the simulation (as links_lgb_a1); mode real: simulation anchoring drawn per limb at the test's
  anchored shares (unanchored test tiles then look like training ones).
-> links_test_<fit><suffix>.npz (test_succ, test_score, oof_succ, oof_score), links_oof_<fit>_<mode>.npz, log lines RESULT/SELFCHECK/TEST."""
import os, sys, time, argparse, json
import numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
from tdlib import *

ap = argparse.ArgumentParser(); ap.add_argument("--fit", default="K7"); ap.add_argument("--mode", default="spec", choices=["spec", "real", "aug", "deaug"])
ap.add_argument("--selfcheck", default="6,18,20"); ap.add_argument("--members", type=int, default=8); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--W", type=int, default=16); ap.add_argument("--topk", type=int, default=50)
a = ap.parse_args()
T0 = time.time()
TAG = f"{a.fit}_{a.mode}"
def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


FIT = FITS[a.fit]
S = load_stage(FIT); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]; fold = S["oof_fold"]; N = len(sbj)
st = np.load(os.path.join(FIT, "stage.npz"), allow_pickle=True); tsbj = st["test_sbj"].astype(np.int64); tsens = st["sensor_test"].astype(np.int64)
NT = len(tsbj)
LZP = os.path.join(FIT, "link_logodds.npz")
REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]
fold_of = {int(s): int(f) for s, f in zip(sbj, fold)}
subs = [int(x) for x in np.unique(sbj)]
TI = np.load(os.path.join(CACHE, "test_inputs.npz"))
assert (TI["drow"] == np.arange(NT)).all()
TESTCHAIN = "testdeaug" if a.mode == "deaug" else "test"                # chains.py (--deaug) cache prefix of the real test
if a.mode == "spec":
    RATES = np.ones(4)
elif a.mode == "deaug":
    RATES = np.array([np.load(os.path.join(CACHE, "deaug_inputs.npz"))["anch"][TI["drow"]][tsens == L].mean() for L in range(4)])
else:
    RATES = np.array([TI["anch_d"][TI["drow"]][tsens == L].mean() for L in range(4)])
log(f"fit {a.fit} mode {a.mode}: simulation anchor rate per pipeline limb {np.round(RATES, 4).tolist()}")


# ------------------------------------------------------------------ training tables (simulation, common node set)
def sim_table(s):
    loc = np.flatnonzero(sbj == s); n = len(loc)
    rows = loc[np.lexsort((S["oof_start"][loc], S["oof_rec"][loc]))]
    p_of = np.full(N, -1); p_of[rows] = np.arange(n); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
    pos = p_of[loc]; lim = sens[loc]
    if a.mode in ("aug", "deaug"):                        # arm chains rebuilt on augmented tiles; own tiles at augmented seconds unanchored
        z = np.load(os.path.join(CACHE, f"chain{a.mode}_s{s}.npz")); assert (z["rows"] == rows).all()
        anch = ~z["aug"][lim, pos]
    else:
        anch = np.random.default_rng(s).random(n) < RATES[lim]
        z = np.load(os.path.join(FEAS, "cache", f"chain_lgb_s{s}.npz")); assert (z["rows"] == rows).all()
    owner = np.full((4, n), -1, np.int64); k = np.flatnonzero(anch); owner[lim[k], pos[k]] = k
    tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
    chains = [chain_tuple(z[f"succ{L}"], z[f"conf{L}"]) for L in range(4)]
    LZ = np.load(LZP); cand, Lo = topk_cands(LZ[f"oof_{s}_cand"], LZ[f"oof_{s}_L"], a.topk)
    t = subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, tl, a.W); t["loc"] = loc
    return t


tabs = {t["s"]: t for t in Parallel(n_jobs=a.jobs)(delayed(sim_table)(s) for s in sorted(subs, key=lambda s: -(sbj == s).sum()))}
names = tabs[subs[0]]["names"]
X = np.concatenate([tabs[s]["X"] for s in subs]); y = np.concatenate([tabs[s]["y"] for s in subs])
fo = np.concatenate([np.full(len(tabs[s]["y"]), fold_of[s]) for s in subs])
log(f"training tables: {len(y)} pairs, positives {y.sum()} ({y.mean():.4f}); features {names}")
rk = X[:, names.index("rk_row")]; LO = X[:, names.index("lo")].astype(np.float64)
rng = np.random.default_rng(0); keep_all = (y == 1) | (rk < 10); samp = rng.random(len(y)) < 0.2
trm = keep_all | samp; wts = np.where(keep_all, 1.0, 5.0)
COLS1 = feat_cols(names); NAMES2 = names + GAP_NAMES; COLS2 = feat_cols(NAMES2)


def nested_and_full(Xf, cols, base, label):
    """5 nested fold models (OOF prediction) + one model on all training subjects; models saved"""
    pf = os.path.join(CACHE, f"pred_{TAG}_{label}.npy")
    paths = [os.path.join(MODELS, f"{TAG}_{label}_f{f}.txt") for f in range(5)] + [os.path.join(MODELS, f"{TAG}_{label}_full.txt")]
    if os.path.exists(pf) and all(os.path.exists(p) for p in paths):
        log(f"{label}: cached"); return np.load(pf)
    Xm = Xf[:, cols]; pred = np.zeros(len(y), np.float32)
    for f in range(5):
        tr = (fo != f) & trm; te = fo == f
        m = lgb.train(P_LINK, lgb.Dataset(Xm[tr], y[tr], weight=wts[tr], init_score=base[tr]), ROUNDS_LINK); m.save_model(paths[f])
        pred[te] = m.predict(Xm[te], raw_score=True) + base[te]
        if f == 0:
            imp = m.feature_importance("gain"); nm_ = [([*names, *GAP_NAMES])[k] for k in cols]
            log(f"{label} top features:", [(nm_[k], round(float(imp[k] / imp.sum()), 3)) for k in np.argsort(-imp)[:12]])
    m = lgb.train(P_LINK, lgb.Dataset(Xm[trm], y[trm], weight=wts[trm], init_score=base[trm]), ROUNDS_LINK); m.save_model(paths[5])
    np.save(pf, pred); log(f"{label}: nested + full models done")
    return pred


def split(pred, keys=None):
    off = 0; prs = {}
    for s in subs:
        k = len(tabs[s]["y"]); prs[s] = pred[off:off + k]; off += k
    return prs


def oof_match(pred, members, label):
    prs = split(pred)
    out = Parallel(n_jobs=a.jobs)(delayed(match)(tabs[s], prs[s], members, s) for s in subs)
    Su = np.full((members, N), -1, np.int64); Sc = np.full((members, N), -50.0, np.float32); loc_su = {}
    for s, res in out:
        loc = tabs[s]["loc"]; loc_su[s] = res[0][0]
        for k, (su, sc) in enumerate(res):
            Su[k, loc] = np.where(su >= 0, loc[np.maximum(su, 0)], -1); Sc[k, loc] = sc
    h = ts >= 0; e = Su[0][h] == ts[h]; same = (sens == sens[np.maximum(ts, 0)])[h]
    log(f"RESULT {label} {TAG}: OOF exact successor {e.mean():.4f} | same-limb {e[same].mean():.4f} cross-limb {e[~same].mean():.4f} | by fold "
        + " ".join(f"{np.mean(e[fold[h] == f_]):.4f}" for f_ in range(5)))
    log("   OOF exact by limb of A:", " ".join(f"{SENS[L]} {np.mean(e[sens[h] == L]):.4f}" for L in range(4)))
    return Su, Sc, loc_su


pred1 = nested_and_full(X, COLS1, LO, "s1")
Su1, _, su1 = oof_match(pred1, 1, "stage1")
G = Parallel(n_jobs=a.jobs)(delayed(gap_feats)(tabs[s], su1[s], a.W) for s in subs)
Xk = np.concatenate([X, np.concatenate(G), pred1[:, None]], 1); del G
pred2 = nested_and_full(Xk, COLS2, pred1.astype(np.float64), "s2"); del Xk
Su, Sc, _ = oof_match(pred2, a.members, "stage2")
for k in range(a.members):
    Sc[k] = qnorm(Su[k], Sc[k], sbj, REF)
h = ts >= 0
log(f"   perturbed mean exact {np.mean([(Su[k][h] == ts[h]).mean() for k in range(1, a.members)]):.4f}")
np.savez(os.path.join(HERE, f"links_oof_{TAG}.npz"), oof_succ=Su, oof_score=Sc)
OOF_SU, OOF_SC = Su, Sc
del X


# ------------------------------------------------------------------ apply to a 'test' world (real or fake) subject
def load_models(which):
    if which == "full":
        return [lgb.Booster(model_file=os.path.join(MODELS, f"{TAG}_{l}_full.txt")) for l in ("s1", "s2")]
    return [lgb.Booster(model_file=os.path.join(MODELS, f"{TAG}_{l}_f{which}.txt")) for l in ("s1", "s2")]


def world_table(cf, s, cand, Lo, tl, rates_rng=None):
    """table of one subject from a chains.py cache file (per-limb node sets). For the self-check in mode 'real' the
    anchoring is thinned at the simulation rates (unanchored tiles get pos 0 and no owner, exactly like the real test)."""
    lim = cf["lim"].astype(np.int64); n = len(lim); anch = cf["anch"].astype(bool).copy(); pos = cf["pos"].astype(np.int64).copy()
    owner = cf["owner"].astype(np.int64).copy()
    if rates_rng is not None:
        anch &= rates_rng.random(n) < RATES[lim]
    k = np.flatnonzero(~anch); pos[k] = 0
    owner = np.where(owner >= 0, owner, -1); keep = np.zeros_like(owner, bool)
    ai = np.flatnonzero(anch); keep[lim[ai], cf["pos"][ai]] = True; owner = np.where(keep, owner, -1)
    assert ((owner >= 0).sum() == anch.sum())
    for i in ai[:50]:
        assert owner[lim[i], pos[i]] == i
    chains = [chain_tuple(cf[f"succ{L}"], cf[f"conf{L}"]) for L in range(4)]
    cand, Lo = topk_cands(cand, Lo, a.topk)
    return subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, tl, a.W)


def score_world(t, models, s, members):
    m1, m2 = models
    base = t["X"][:, names.index("lo")].astype(np.float64)
    p1 = (m1.predict(t["X"][:, COLS1], raw_score=True) + base).astype(np.float32)
    _, r1 = match(t, p1, 1, s)
    g = gap_feats(t, r1[0][0], a.W)
    Xk_ = np.concatenate([t["X"], g, p1[:, None]], 1)
    p2 = (m2.predict(Xk_[:, COLS2], raw_score=True) + p1).astype(np.float64).astype(np.float32)
    _, res = match(t, p2, members, s)
    return res, r1[0][0]


# ------------------------------------------------------------------ (5) self-check on fake 2025 worlds
SC = [int(x) for x in a.selfcheck.split(",")] if a.selfcheck else []
own = np.load(os.path.join(FIT, "links.npz"))
sc_out = {}
if SC:
    LZ = np.load(LZP); ee_new, ee_own, ee_sim, ee_s1 = [], [], [], []
    for s in SC:
        cf = np.load(os.path.join(CACHE, f"fake_chain_s{s}.npz"))
        loc = np.flatnonzero(sbj == s); n = len(loc); l_of = np.full(N, -1); l_of[loc] = np.arange(n)
        assert (cf["lim"] == sens[loc]).all()
        tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
        t = world_table(cf, s, LZ[f"oof_{s}_cand"], LZ[f"oof_{s}_L"], tl, np.random.default_rng(s) if a.mode == "real" else None)
        res, s1 = score_world(t, load_models(fold_of[s]), s, 1)
        su = res[0][0]; hh = tl >= 0
        e_new = su[hh] == tl[hh]; e_s1 = s1[hh] == tl[hh]
        o = own["oof_succ"][0][loc]; e_own = o[hh] == loc[tl[hh]]
        e_sim = OOF_SU[0][loc][hh] == loc[tl[hh]]
        ee_new.append(e_new); ee_own.append(e_own); ee_sim.append(e_sim); ee_s1.append(e_s1)
        log(f"SELFCHECK sbj {s} (fold {fold_of[s]}): exact successor fake-2025 path {e_new.mean():.4f} (stage1 {e_s1.mean():.4f}) | nested sim links {e_sim.mean():.4f} | fit's own links {e_own.mean():.4f}")
    sc_out = dict(selfcheck_exact_new=float(np.concatenate(ee_new).mean()), selfcheck_exact_own=float(np.concatenate(ee_own).mean()),
                  selfcheck_exact_sim=float(np.concatenate(ee_sim).mean()), selfcheck_exact_stage1=float(np.concatenate(ee_s1).mean()))
    log("SELFCHECK pooled:", json.dumps({k: round(v, 4) for k, v in sc_out.items()}))


# ------------------------------------------------------------------ (4)/(6) real test
LZ = np.load(LZP); models = load_models("full")
TSu = np.full((a.members, NT), -1, np.int64); TSc = np.full((a.members, NT), -50.0, np.float32)
stats = {}; agree_n = 0; agree_k = 0
for s in (22, 23, 24, 25):
    cf = np.load(os.path.join(CACHE, f"{TESTCHAIN}_chain_s{s}.npz"))
    loc = np.flatnonzero(tsbj == s); n = len(loc)
    assert (cf["loc"] == loc).all() and (cf["lim"] == tsens[loc]).all()
    t = world_table(cf, s, LZ[f"test_{s}_cand"], LZ[f"test_{s}_L"], np.full(n, -1))
    res, _ = score_world(t, models, s, a.members)
    for k, (su, sc) in enumerate(res):
        TSu[k, loc] = np.where(su >= 0, loc[np.maximum(su, 0)], -1); TSc[k, loc] = sc
    # agreement of same-limb links (anchored both ends) with the 2025 chain of that limb
    su = res[0][0]; lim = t["lim"]; an = t["anch"]; pos = t["pos"]
    m = (su >= 0) & (lim == lim[np.maximum(su, 0)]) & an & an[np.maximum(su, 0)]
    i = np.flatnonzero(m); ch_next = np.array([cf[f"succ{lim[x]}"][pos[x]] for x in i]) if len(i) else np.zeros(0)
    agree_n += len(i); agree_k += int(np.sum(ch_next == pos[su[i]]))
    log(f"TEST sbj {s}: n={n} linked {np.mean(su >= 0):.3f}, pairs {len(t['pi'])}")
for k in range(a.members):
    TSc[k] = qnorm(TSu[k], TSc[k], tsbj, REF)
ol = own["test_succ"][0]
changed = float(np.mean(TSu[0] != ol)); same_new = float(np.mean(tsens[np.maximum(TSu[0], 0)][TSu[0] >= 0] == tsens[TSu[0] >= 0]))
same_own = float(np.mean(tsens[np.maximum(ol, 0)][ol >= 0] == tsens[ol >= 0]))
# agreement for the fit's own links too
own_agree_n = own_agree_k = 0
for s in (22, 23, 24, 25):
    cf = np.load(os.path.join(CACHE, f"{TESTCHAIN}_chain_s{s}.npz")); loc = np.flatnonzero(tsbj == s); l_of = np.full(NT, -1); l_of[loc] = np.arange(len(loc))
    su = np.where(ol[loc] >= 0, l_of[np.maximum(ol[loc], 0)], -1); lim = cf["lim"]; an = cf["anch"].astype(bool); pos = cf["pos"]
    m = (su >= 0) & (lim == lim[np.maximum(su, 0)]) & an & an[np.maximum(su, 0)]; i = np.flatnonzero(m)
    own_agree_n += len(i); own_agree_k += int(sum(cf[f"succ{lim[x]}"][pos[x]] == pos[su[x]] for x in i))
stats = dict(test_changed_share=changed, same_limb_share_new=same_new, same_limb_share_own=same_own,
             chain_agree_new=agree_k / max(agree_n, 1), chain_agree_new_n=agree_n, chain_agree_own=own_agree_k / max(own_agree_n, 1), chain_agree_own_n=own_agree_n,
             linked_new=float(np.mean(TSu[0] >= 0)), perturbed_changed=float(np.mean([np.mean(TSu[k] != TSu[0]) for k in range(1, a.members)])))
stats["changed_by_limb"] = {SENS[L]: float(np.mean((TSu[0] != ol)[tsens == L])) for L in range(4)}
own_oof = own["oof_succ"][0]
stats["oof_changed_by_limb"] = {SENS[L]: float(np.mean((OOF_SU[0] != own_oof)[sens == L])) for L in range(4)}
log("TEST stats:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}))

# outputs
if a.mode == "spec" and a.fit == "K7":
    sim = np.load(os.path.join(FEAS, "cache", "links_lgb_a1.npz")); oof_su, oof_sc = sim["oof_succ"], sim["oof_score"]; oof_src = "research2 links_lgb_a1.npz"
    h = ts >= 0; log(f"   cached links_lgb_a1 vs this run's nested OOF: first matching identical {np.mean(oof_su[0] == OOF_SU[0]):.4f}, exact {np.mean(oof_su[0][h] == ts[h]):.4f} vs {np.mean(OOF_SU[0][h] == ts[h]):.4f}")
elif a.mode == "spec" and a.fit == "K9" and os.path.exists(os.path.join(FEAS, "cache", "links_lgb_a1_k9.npz")):
    sim = np.load(os.path.join(FEAS, "cache", "links_lgb_a1_k9.npz")); oof_su, oof_sc = sim["oof_succ"], sim["oof_score"]; oof_src = "research2 links_lgb_a1_k9.npz"
else:
    oof_su, oof_sc = OOF_SU, OOF_SC; oof_src = f"this run's nested OOF (links_oof_{TAG}.npz)"
out = os.path.join(HERE, f"links_test_{a.fit}{'' if a.mode == 'spec' else '_' + a.mode}.npz")
np.savez(out, test_succ=TSu, test_score=TSc, oof_succ=oof_su, oof_score=oof_sc)
json.dump(dict(fit=a.fit, mode=a.mode, rates=RATES.tolist(), oof_source=oof_src, **sc_out, **stats), open(out[:-4] + "_info.json", "w"), indent=1)
log(f"wrote {out} (oof from {oof_src})")
