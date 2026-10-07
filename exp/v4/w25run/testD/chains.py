"""Task D, steps (1)+(2): mappings with asserts, per-limb 2025 chains for the real test subjects (anchored twins replaced by
our clean tiles), scored by the train_chain.py LightGBM retrained on ALL 22 training subjects; plus a fake 2025 world for
the self-check subjects (their 4-limb sim tiles, rows of every limb shuffled independently) scored by the model trained on
the other folds.
  python chains.py [--selfcheck 6,18,20]
-> models/chain_full.txt, models/chain_f{f}.txt, cache/test_chain_s{s}.npz, cache/fake_chain_s{s}.npz, cache/test_inputs.npz"""
import os, sys, time, glob, csv, argparse
import numpy as np
from joblib import Parallel, delayed
from tdlib import *

ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", default="6,18,20"); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--deaug", action="store_true")
a = ap.parse_args()
T0 = time.time()
def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


S = load_stage(K7); fold_of = {int(s): int(f) for s, f in zip(S["oof_sbj"], S["oof_fold"])}
SC = [int(x) for x in a.selfcheck.split(",")] if a.selfcheck else []
SC_FOLDS = sorted({fold_of[s] for s in SC})


# ------------------------------------------------------------------ chain scorer: full + the self-check fold(s)
def train_chain_models():
    import lightgbm as lgb
    need = [("full", os.path.join(MODELS, "chain_full.txt"))] + [(f, os.path.join(MODELS, f"chain_f{f}.txt")) for f in SC_FOLDS]
    if all(os.path.exists(p) for _, p in need):
        log("chain models cached"); return
    subs = sorted(int(f.split("_s")[-1][:-4]) for f in glob.glob(os.path.join(FEAS, "cache", "feat_s*.npz")))
    assert subs == list(range(22)), subs
    Xs, ys, ss = [], [], []
    for s in subs:
        z = np.load(os.path.join(FEAS, "cache", f"feat_s{s}.npz"))
        for L in range(4):
            X = z[f"X{L}"]; Xs.append(X); ys.append(z[f"y{L}"]); ss.append(np.full(len(X), s))
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.int8); sb = np.concatenate(ss); fo = np.array([fold_of[int(s)] for s in subs])[sb]
    del Xs
    rr, rc = X[:, FEATS.index("rank_row")], X[:, FEATS.index("rank_col")]
    rng = np.random.default_rng(0); train_mask = (y == 1) | (rr < 4) | (rc < 4) | (rng.random(len(y)) < 0.05)
    log(f"chain training pool {len(y)} pairs, {train_mask.sum()} kept")
    for f, p in need:
        if os.path.exists(p):
            continue
        tr = train_mask if f == "full" else (fo != f) & train_mask
        m = lgb.train(P_CHAIN, lgb.Dataset(X[tr], y[tr]), ROUNDS_CHAIN); m.save_model(p)
        log(f"chain model {f}: trained on {tr.sum()} rows")
        if f != "full":                                  # replication check against the cached nested predictions of train_chain.py
            te = fo == f; pr = m.predict(X[te], raw_score=True); ref = np.load(os.path.join(FEAS, "cache", f"chain_lgb_pred_f{f}.npy"))
            log(f"   fold {f} vs cached train_chain predictions: corr {np.corrcoef(pr, ref)[0, 1]:.6f} max|diff| {np.abs(pr - ref).max():.4f}")


# ------------------------------------------------------------------ real test world: mappings with asserts
def real_world():
    A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
    meta = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
    id26 = np.array([int(r["id"]) for r in meta]); s26 = np.array([int(r["sbj_id"]) for r in meta])
    sens26 = np.array([SENS.index(r["sensor_location"]) for r in meta])
    assert (id26 == np.arange(len(id26))).all() and len(A26) == len(id26)
    st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True)
    pid = np.array([int(x) for x in st["ids"]])                                     # pipeline row -> test id
    row_of_id = np.full(id26.max() + 1, -1); row_of_id[id26] = np.arange(len(id26)); drow = row_of_id[pid]   # pipeline row -> data row
    assert (drow >= 0).all() and len(np.unique(drow)) == len(drow) == len(A26)
    assert (st["test_sbj"] == s26[drow]).all() and (st["sensor_test"] == sens26[drow]).all()
    log(f"pipeline <-> data rows: identity {bool((drow == np.arange(len(drow))).all())}; sbj and sensor agree 100%")
    z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; s25 = z["sbj"]; l25 = z["limb"]
    m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]; corr = m["corr"]; margin = m["margin"]
    dist = np.abs(A26.astype(np.float64) - A25[twin].astype(np.float64)).max((1, 2))
    exact = dist < 1e-4
    pl25 = MAP25[l25]
    ok_l = pl25[twin[exact]] == sens26[exact]; ok_s = s25[twin[exact]] == s26[exact]
    log(f"exact twins {exact.mean():.4f}; limb encoding check on exact twins: sensor == MAP25[limb25] {ok_l.mean():.4f}, sbj {ok_s.mean():.4f}")
    assert ok_l.all() and ok_s.all()
    for c in range(4):                                     # also: the encoding is the ONLY consistent one
        alt = [np.mean(l25[twin[exact]][sens26[exact] == L] == c) for L in range(4)]
        assert max(alt) > 0.999 and alt[MAP25[c]] > 0.999
    conf_ne = ~exact & (corr > 0.99) & (margin > 0.05)
    anch = exact | conf_ne
    assert (pl25[twin[anch]] == sens26[anch]).all() and (s25[twin[anch]] == s26[anch]).all()
    assert len(np.unique(twin[anch])) == anch.sum()
    log(f"anchored {anch.mean():.4f} (exact {exact.mean():.4f} + confident non-exact {conf_ne.mean():.4f}); per pipeline limb:",
        {SENS[L]: round(float(anch[sens26 == L].mean()), 4) for L in range(4)})
    p_sbj = st["test_sbj"].astype(np.int64); p_sens = st["sensor_test"].astype(np.int64)
    return dict(A26=A26, drow=drow, p_sbj=p_sbj, p_sens=p_sens, A25=A25, s25=s25, pl25=pl25, twin=twin, anch_d=anch, exact_d=exact)


# ------------------------------------------------------------------ fake world from training subjects (self-check)
def fake_world(subs):
    parts26, sb26, se26, true_succ_loc = [], [], [], {}
    A25s, s25s, l25s = [], [], []
    inv25 = np.argsort(MAP25)                                                      # pipeline limb -> 2025 code
    twin_parts = []; off25 = 0; truechain = {}; posold = []; nxts = {}
    for s in subs:
        ii, T = subject_tiles(S, s); n = len(ii)                                  # positions (rec, start) order
        loc = np.flatnonzero(S["oof_sbj"] == s)                                  # link-local (pipeline) order
        p_of = np.full(len(S["oof_y"]), -1); p_of[ii] = np.arange(n); pos_true = p_of[loc]
        lim = S["sensor_oof"][loc]
        parts26.append(T[lim, pos_true]); sb26.append(np.full(n, s)); se26.append(lim)
        rng = np.random.default_rng(777 + s); tw = np.full(n, -1, np.int64)
        rec = S["oof_rec"][ii]; nxts[s] = np.where(np.r_[rec[1:] == rec[:-1], False], np.arange(1, n + 1), -1)   # true next position
        for L in range(4):
            perm = rng.permutation(n); inv = np.argsort(perm)                     # block row r holds position perm[r]
            A25s.append(T[L][perm]); s25s.append(np.full(n, s)); l25s.append(np.full(n, inv25[L])); posold.append(perm)
            k = lim == L; tw[k] = off25 + inv[pos_true[k]]
            off25 += n
        twin_parts.append(tw)
    A26 = np.concatenate(parts26); A25 = np.concatenate(A25s); s25 = np.concatenate(s25s); l25 = np.concatenate(l25s)
    twin = np.concatenate(twin_parts); posold = np.concatenate(posold)
    g = np.random.default_rng(99).permutation(len(A25)); gi = np.argsort(g)       # interleave all 2025 rows (as the real file)
    A25, s25, l25 = A25[g], s25[g], l25[g]; twin = gi[twin]; posnew = posold[g]   # new row i holds old row g[i]
    for s in subs:                                                                 # true chain successor in prepare_subject's row order
        truechain[s] = {}
        for L in range(4):
            r = np.flatnonzero((s25 == s) & (MAP25[l25] == L)); p = posnew[r]; idx = np.argsort(p)   # idx[position] = chain index
            nx = nxts[s][p]; truechain[s][L] = np.where(nx >= 0, idx[np.maximum(nx, 0)], -1)
    p_sbj = np.concatenate(sb26); p_sens = np.concatenate(se26)
    dist = np.abs(A26.astype(np.float64) - A25[twin].astype(np.float64)).max((1, 2)); exact = dist < 1e-4
    assert exact.all() and (MAP25[l25[twin]] == p_sens).all()
    return dict(A26=A26, drow=np.arange(len(A26)), p_sbj=p_sbj, p_sens=p_sens, A25=A25, s25=s25, pl25=MAP25[l25], twin=twin,
                anch_d=exact, exact_d=exact, truechain=truechain, g=g)


def subject_inputs(wd, s):
    loc = np.flatnonzero(wd["p_sbj"] == s); d = wd["drow"][loc]
    lim = wd["p_sens"][loc]; anch = wd["anch_d"][d]; tw = np.where(anch, wd["twin"][d], -1)
    pre = prepare_subject(s, wd["A26"][d], lim, tw, anch, wd["A25"], wd["s25"], wd["pl25"])
    return loc, lim, anch, pre


def run_world(wd, subs, model_file, tag):
    jobs, meta = [], {}
    for s in subs:
        loc, lim, anch, pre = subject_inputs(wd, s); meta[s] = (loc, lim, anch, pre)
        for L in range(4):
            jobs.append((s, L, pre["Ts"][L]))
    jobs.sort(key=lambda j: -len(j[2]))
    res = Parallel(n_jobs=a.jobs)(delayed(score_chain_limb)(T, L, model_file) for s, L, T in jobs)
    out = {}
    for (s, L, _), (L_, su, cf) in zip(jobs, res):
        assert L == L_; out[(s, L)] = (su, cf)
    for s in subs:
        loc, lim, anch, pre = meta[s]
        d = dict(loc=loc, lim=lim, anch=anch, pos=pre["pos"], owner=pre["owner"], exact=wd["exact_d"][wd["drow"][loc]])
        for L in range(4):
            d[f"succ{L}"], d[f"conf{L}"] = out[(s, L)]; d[f"rows{L}"] = pre["rows_L"][L]
        msg = " ".join(f"L{L} linked {np.mean(d[f'succ{L}'] >= 0):.3f} conf {np.mean(d[f'conf{L}']):.3f}" for L in range(4))
        if "truechain" in wd:
            tc = wd["truechain"][s]
            msg += " | exact chain " + " ".join(f"L{L} {np.mean(d[f'succ{L}'][tc[L] >= 0] == tc[L][tc[L] >= 0]):.3f}" for L in range(4))
            for L in range(4):
                d[f"true{L}"] = tc[L]
        np.savez(os.path.join(CACHE, f"{tag}_chain_s{s}.npz"), **d)
        log(f"{tag} sbj {s} n={len(loc)}: {msg}")


def deaug_world(wd):
    """the same real world with the de-augmented 2025 array and the extended twins of deaug.py"""
    d = np.load(os.path.join(CACHE, "deaug_inputs.npz"))
    tw, an = d["twin"], d["anch"]
    assert (tw[wd["exact_d"]] == wd["twin"][wd["exact_d"]]).all() and an[wd["exact_d"]].all()
    ce = wd["anch_d"] & ~wd["exact_d"]                                  # the old 'confident non-exact' twins vs the exact ones now
    log(f"deaug: old confident non-exact twins {ce.sum()}, re-found identically {np.mean(tw[ce] == wd['twin'][ce]):.4f}")
    A25 = d["A25"]
    a_idx = np.flatnonzero(an)
    dist = np.abs(wd["A26"][a_idx].astype(np.float64) - A25[tw[a_idx]].astype(np.float64)).max((1, 2))
    log(f"deaug: anchored {an.mean():.4f}; de-augmented twin rows vs our tiles max|diff| q50/q99/max {np.quantile(dist, 0.5):.4f} {np.quantile(dist, 0.99):.4f} {dist.max():.4f}")
    out = dict(wd); out.update(A25=A25, twin=np.where(an, tw, -1), anch_d=an)
    return out


if __name__ == "__main__":
    ap2 = argparse.ArgumentParser(); ap2.add_argument("--deaug", action="store_true")
    deaug = "--deaug" in sys.argv
    train_chain_models()
    wd = real_world()
    if deaug:
        wd = deaug_world(wd)
        run_world(wd, [22, 23, 24, 25], os.path.join(MODELS, "chain_full.txt"), "testdeaug")
        log("done"); sys.exit(0)
    np.savez(os.path.join(CACHE, "test_inputs.npz"), anch_d=wd["anch_d"], exact_d=wd["exact_d"], drow=wd["drow"])
    if SC:
        fw = fake_world(SC)
        for f in SC_FOLDS:
            run_world(fw, [s for s in SC if fold_of[s] == f], os.path.join(MODELS, f"chain_f{f}.txt"), "fake")
        log("fake world done")
    run_world(wd, [22, 23, 24, 25], os.path.join(MODELS, "chain_full.txt"), "test")
    log("done")
