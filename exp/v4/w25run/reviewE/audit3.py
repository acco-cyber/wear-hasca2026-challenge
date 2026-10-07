"""Review E, part 3: re-derive stored outputs from cached inputs.
 (i)  test path: rebuild the test table of one subject from the cached real-test chains + the saved FULL models, decode the 8
      matchings, qnorm, and compare with the stored links_test_<fit>[_deaug].npz rows of that subject (succ and score).
 (ii) self-check: fake-2025 world of a fold-4 subject through the test code path with the fold-4 spec models must equal the
      nested simulation links stored in links_oof_<fit>_spec.npz.
  python audit3.py --fit K7 --mode deaug --sbj 25 [--selfcheck 20]"""
import os, sys, time, argparse
import numpy as np
TD = r"E:\Claude code\wear\exp\v4\w25run\testD"
sys.path.insert(0, TD)
from tdlib import *                                                   # noqa: E402
import lightgbm as lgb                                                # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--fit", default="K7"); ap.add_argument("--mode", default="deaug")
ap.add_argument("--sbj", default="25"); ap.add_argument("--selfcheck", default="")
a = ap.parse_args(); T0 = time.time()
FIT = FITS[a.fit]; TAG = f"{a.fit}_{a.mode}"
st = np.load(os.path.join(FIT, "stage.npz"), allow_pickle=True); tsbj = st["test_sbj"].astype(np.int64); tsens = st["sensor_test"].astype(np.int64)
LZ = np.load(os.path.join(FIT, "link_logodds.npz")); REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]


def my_world_table(cf, cand, Lo, tl):
    lim = cf["lim"].astype(np.int64); n = len(lim); anch = cf["anch"].astype(bool)
    pos = np.where(anch, cf["pos"], 0).astype(np.int64)
    owner = np.full((4, n), -1, np.int64); ai = np.flatnonzero(anch); owner[lim[ai], pos[ai]] = ai
    assert np.array_equal(owner, np.where(cf["owner"] >= 0, cf["owner"], -1))
    chains = [chain_tuple(cf[f"succ{L}"], cf[f"conf{L}"]) for L in range(4)]
    cand, Lo = topk_cands(cand, Lo, 50)
    return subject_table(0, n, lim, anch, pos, owner, chains, cand, Lo, tl, 16)


def decode(t, m1, m2, s, members):
    names = t["names"]; c1 = feat_cols(names); c2 = feat_cols(names + GAP_NAMES)
    base = t["X"][:, names.index("lo")].astype(np.float64)
    p1 = (m1.predict(t["X"][:, c1], raw_score=True) + base).astype(np.float32)
    _, r1 = match(t, p1, 1, s); g = gap_feats(t, r1[0][0], 16)
    Xk = np.concatenate([t["X"], g, p1[:, None]], 1)
    p2 = (m2.predict(Xk[:, c2], raw_score=True) + p1).astype(np.float64).astype(np.float32)
    _, res = match(t, p2, members, s)
    return res


if a.sbj:
    stored = np.load(os.path.join(TD, f"links_test_{a.fit}{'' if a.mode == 'spec' else '_' + a.mode}.npz"))
    m1 = lgb.Booster(model_file=os.path.join(TD, "models", f"{TAG}_s1_full.txt")); m2 = lgb.Booster(model_file=os.path.join(TD, "models", f"{TAG}_s2_full.txt"))
    pref = "testdeaug" if a.mode == "deaug" else "test"
    for s in [int(x) for x in a.sbj.split(",")]:
        cf = np.load(os.path.join(TD, "cache", f"{pref}_chain_s{s}.npz")); loc = np.flatnonzero(tsbj == s); n = len(loc)
        assert (cf["loc"] == loc).all() and (cf["lim"] == tsens[loc]).all()
        t = my_world_table(cf, LZ[f"test_{s}_cand"], LZ[f"test_{s}_L"], np.full(n, -1))
        res = decode(t, m1, m2, s, 8)
        eq_su, eq_sc = [], []
        for k, (su, sc) in enumerate(res):
            g = np.where(su >= 0, loc[np.maximum(su, 0)], -1)
            eq_su.append(float(np.mean(g == stored["test_succ"][k][loc])))
            q = qnorm(g, sc, np.full(n, s), REF)
            eq_sc.append(float(np.abs(q - stored["test_score"][k][loc]).max()))
        print(f"[{time.time() - T0:.0f}s] TEST-PATH {TAG} sbj {s}: stored succ reproduced per matching {np.round(eq_su, 4).tolist()} | score max|diff| {max(eq_sc):.2e}", flush=True)

if a.selfcheck:
    S = load_stage(FIT); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]; N = len(sbj)
    oof = np.load(os.path.join(TD, f"links_oof_{a.fit}_spec.npz"))["oof_succ"][0]
    m1 = lgb.Booster(model_file=os.path.join(TD, "models", f"{a.fit}_spec_s1_f4.txt")); m2 = lgb.Booster(model_file=os.path.join(TD, "models", f"{a.fit}_spec_s2_f4.txt"))
    for s in [int(x) for x in a.selfcheck.split(",")]:
        cf = np.load(os.path.join(TD, "cache", f"fake_chain_s{s}.npz")); loc = np.flatnonzero(sbj == s); n = len(loc)
        l_of = np.full(N, -1); l_of[loc] = np.arange(n); tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1)
        assert (cf["lim"] == sens[loc]).all() and cf["anch"].all()
        # the fake chains must be the cached simulation chains relabelled: true chain accuracy per limb
        t = my_world_table(cf, LZ[f"oof_{s}_cand"], LZ[f"oof_{s}_L"], tl)
        su = decode(t, m1, m2, s, 1)[0][0]; g = np.where(su >= 0, loc[np.maximum(su, 0)], -1)
        h = tl >= 0
        print(f"[{time.time() - T0:.0f}s] SELFCHECK {a.fit} sbj {s}: fake-2025 test path exact {np.mean(su[h] == tl[h]):.4f}; identical to nested sim links {np.mean(g == oof[loc]):.4f}; "
              f"sim exact {np.mean(oof[loc][h] == loc[tl[h]]):.4f}; fake chain exact " + " ".join(f"{np.mean(cf[f'succ{L}'][cf[f'true{L}'] >= 0] == cf[f'true{L}'][cf[f'true{L}'] >= 0]):.3f}" for L in range(4)), flush=True)
