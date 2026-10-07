"""Review E, part 1: independent label-free checks of the testD test-side link pipeline (read only).
 A mappings (pipeline row <-> data row, w25 limb codes by an UNRESTRICTED exact-hash twin search)
 B test chain caches (rows per limb, pos/owner/twin consistency, chains are partial permutations)
 C output link files (one-to-one, within subject, changed share, same-limb share, chain agreement), plus the same chain
   agreement measured on the OOF simulation links for comparison."""
import os, sys, csv, json
import numpy as np
W = r"E:\Claude code\wear"; TD = os.path.join(W, "exp", "v4", "w25run", "testD")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4"); K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
SENS = ["right_arm", "right_leg", "left_leg", "left_arm"]; L25 = ["left_arm", "left_leg", "right_arm", "right_leg"]
MAP25 = np.array([SENS.index(x) for x in L25])
out = {}

# ---------------------------------------------------------------- A mappings
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy"))
meta = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
id26 = np.array([int(r["id"]) for r in meta]); s26 = np.array([int(r["sbj_id"]) for r in meta]); se26 = np.array([SENS.index(r["sensor_location"]) for r in meta])
for nm, d in (("K7", K7), ("K9", K9)):
    st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True)
    pid = np.array([int(x) for x in st["ids"]]); row = {v: i for i, v in enumerate(id26)}; drow = np.array([row[v] for v in pid])
    print(f"A {nm}: pipeline->data identity {bool((drow == np.arange(len(drow))).all())}, sbj agree {np.mean(st['test_sbj'] == s26[drow]):.4f}, "
          f"sensor agree {np.mean(st['sensor_test'] == se26[drow]):.4f}, n {len(pid)}")
    assert (drow == np.arange(len(drow))).all() and (st["test_sbj"] == s26).all() and (st["sensor_test"] == se26).all()
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"]; s25 = z["sbj"]; l25 = z["limb"]
print("A w25 subjects", np.unique(s25).tolist(), "limb codes", np.unique(l25).tolist())
# unrestricted exact twin search: hash the raw float32 bytes of every row
h25 = {}
for i in range(len(A25)):
    h25.setdefault(A25[i].astype(np.float32).tobytes(), []).append(i)
hits = np.full(len(A26), -1); nmulti = 0
for i in range(len(A26)):
    r = h25.get(A26[i].astype(np.float32).tobytes())
    if r is not None:
        hits[i] = r[0]; nmulti += len(r) > 1
ex = hits >= 0
print(f"A unrestricted exact twins {ex.mean():.4f} (multi-hit {nmulti}); sbj agree {np.mean(s25[hits[ex]] == s26[ex]):.4f}")
ct = np.zeros((4, 4), int)
np.add.at(ct, (se26[ex], l25[hits[ex]]), 1)
print("A crosstab pipeline sensor (rows, SENS order) x w25 limb code (cols):\n", ct)
print("A implied map code->pipeline", [int(np.argmax(ct[:, c])) for c in range(4)], "testD MAP25", MAP25.tolist())
assert all(int(np.argmax(ct[:, c])) == MAP25[c] for c in range(4))
for s in np.unique(s26):
    for L in range(4):
        assert np.sum((s25 == s) & (MAP25[l25] == L)) == np.sum(s26 == s)
print("A per (subject, limb) 2025 row counts == our tile counts: OK")
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin0 = m["twin"]
exm = np.abs(A26.astype(np.float64) - A25[twin0].astype(np.float64)).max((1, 2)) < 1e-4
print(f"A match.npz exact twins {exm.mean():.4f}; same rows as hash {np.mean(twin0[ex & exm] == hits[ex & exm]):.4f}; hash-only {np.sum(ex & ~exm)}, match-only {np.sum(exm & ~ex)}")

# ---------------------------------------------------------------- B chain caches
dg = np.load(os.path.join(TD, "cache", "deaug_inputs.npz")); TI = np.load(os.path.join(TD, "cache", "test_inputs.npz"))
A25d = dg["A25"]
for pref, tw_all, an_all, Aref in (("test", np.where(TI["anch_d"], twin0, -1), TI["anch_d"], A25), ("testdeaug", dg["twin"], dg["anch"], A25d)):
    for s in (22, 23, 24, 25):
        cf = np.load(os.path.join(TD, "cache", f"{pref}_chain_s{s}.npz"))
        loc = np.flatnonzero(s26 == s); n = len(loc); lim = cf["lim"]; an = cf["anch"].astype(bool); pos = cf["pos"]; own = cf["owner"]
        assert (cf["loc"] == loc).all() and (lim == se26[loc]).all() and (an == an_all[loc]).all()
        for L in range(4):
            rL = np.flatnonzero((s25 == s) & (MAP25[l25] == L)); assert (cf[f"rows{L}"] == rL).all()
            su = cf[f"succ{L}"]; k = su >= 0
            assert (su[k] < n).all() and (su[k] != np.flatnonzero(k)).all() and np.bincount(su[k], minlength=n).max() <= 1
        ai = np.flatnonzero(an)
        twr = np.array([cf[f"rows{lim[i]}"][pos[i]] for i in ai])
        assert (twr == tw_all[loc[ai]]).all(), (pref, s)
        assert (own[lim[ai], pos[ai]] == ai).all() and (own >= 0).sum() == len(ai)
        dist = np.abs(A26[loc[ai]].astype(np.float64) - Aref[twr].astype(np.float64)).max((1, 2))
        print(f"B {pref} sbj {s}: n {n} anchored {an.mean():.4f}; twin rows == rows{{L}}[pos] OK; owner OK; anchored twin vs our tile max|diff| "
              f"q50 {np.quantile(dist, .5):.4f} q99 {np.quantile(dist, .99):.4f} max {dist.max():.3f}; linked " + " ".join(f"{np.mean(cf[f'succ{L}'] >= 0):.3f}" for L in range(4)))


# ---------------------------------------------------------------- C outputs
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); tsbj = st["test_sbj"]; tsens = st["sensor_test"]; NT = len(tsbj)
sbj = st["oof_sbj"]; ts = st["true_succ"]; sens = st["sensor_oof"]; N = len(sbj); fold = st["oof_fold"]
own = {"K7": np.load(os.path.join(K7, "links.npz")), "K9": np.load(os.path.join(K9, "links.npz"))}


def chain_agree_test(su0, pref, an_all):
    k_ = n_ = 0; per = np.zeros((4, 2))
    for s in (22, 23, 24, 25):
        cf = np.load(os.path.join(TD, "cache", f"{pref}_chain_s{s}.npz")); loc = np.flatnonzero(tsbj == s); lof = np.full(NT, -1); lof[loc] = np.arange(len(loc))
        su = np.where(su0[loc] >= 0, lof[np.maximum(su0[loc], 0)], -1); lim = cf["lim"]; an = an_all[loc]; pos = cf["pos"]
        mm = (su >= 0) & (lim == lim[np.maximum(su, 0)]) & an & an[np.maximum(su, 0)]; i = np.flatnonzero(mm)
        nx = np.array([cf[f"succ{lim[x]}"][pos[x]] for x in i]); ok = nx == pos[su[i]]
        k_ += ok.sum(); n_ += len(i)
        for L in range(4):
            per[L] += [ok[lim[i] == L].sum(), (lim[i] == L).sum()]
    return k_ / n_, n_, (per[:, 0] / np.maximum(per[:, 1], 1)).round(3).tolist()


def chain_agree_oof(su0, mode):
    """same statistic on the OOF simulation: chains over positions (cached nested sim chains)"""
    k_ = n_ = 0; per = np.zeros((4, 2))
    for s in np.unique(sbj):
        loc = np.flatnonzero(sbj == s); n = len(loc); rows = loc[np.lexsort((st["oof_start"][loc], st["oof_rec"][loc]))]
        p_of = np.full(N, -1); p_of[rows] = np.arange(n); pos = p_of[loc]; lim = sens[loc]
        if mode == "deaug":
            zc = np.load(os.path.join(TD, "cache", f"chaindeaug_s{s}.npz")); an = ~zc["aug"][lim, pos]
        else:
            zc = np.load(os.path.join(W, "exp", "v4", "research2", "w25feas", "cache", f"chain_lgb_s{s}.npz")); an = np.ones(n, bool)
        assert (zc["rows"] == rows).all()
        lof = np.full(N, -1); lof[loc] = np.arange(n)
        su = np.where(su0[loc] >= 0, lof[np.maximum(su0[loc], 0)], -1)
        mm = (su >= 0) & (lim == lim[np.maximum(su, 0)]) & an & an[np.maximum(su, 0)]; i = np.flatnonzero(mm)
        nx = np.array([zc[f"succ{lim[x]}"][pos[x]] for x in i]); ok = nx == pos[su[i]]
        k_ += ok.sum(); n_ += len(i)
        for L in range(4):
            per[L] += [ok[lim[i] == L].sum(), (lim[i] == L).sum()]
    return k_ / n_, n_, (per[:, 0] / np.maximum(per[:, 1], 1)).round(3).tolist()


files = {"K7_spec": "links_test_K7.npz", "K9_spec": "links_test_K9.npz", "K7_deaug": "links_test_K7_deaug.npz", "K9_deaug": "links_test_K9_deaug.npz"}
for tag, fn in files.items():
    zf = np.load(os.path.join(TD, fn)); fit = tag[:2]; mode = tag[3:]
    for split, n_, sb in (("test", NT, tsbj), ("oof", N, sbj)):
        su, sc = zf[f"{split}_succ"], zf[f"{split}_score"]; assert su.shape == (8, n_) and sc.shape == (8, n_)
        for k in range(8):
            mk = su[k] >= 0
            assert (su[k][mk] != np.flatnonzero(mk)).all() and (sb[su[k][mk]] == sb[mk]).all() and np.bincount(su[k][mk], minlength=n_).max() <= 1 and np.isfinite(sc[k]).all()
    t0 = zf["test_succ"][0]; o0 = own[fit]["test_succ"][0]; oo = zf["oof_succ"][0]; h = ts >= 0
    same_new = np.mean(tsens[t0[t0 >= 0]] == tsens[t0 >= 0]); same_own = np.mean(tsens[o0[o0 >= 0]] == tsens[o0 >= 0])
    an_all = TI["anch_d"] if mode == "spec" else dg["anch"]; pref = "test" if mode == "spec" else "testdeaug"
    ca_new = chain_agree_test(t0, pref, an_all); ca_own = chain_agree_test(o0, pref, an_all)
    ca_oof = chain_agree_oof(oo, mode); ca_oof_own = chain_agree_oof(own[fit]["oof_succ"][0], mode)
    # changed share per matching and score summaries
    ch = [float(np.mean(zf["test_succ"][k] != own[fit]["test_succ"][k])) for k in range(8)]
    och = float(np.mean(oo != own[fit]["oof_succ"][0]))
    print(f"C {tag}: one-to-one/within-subject/no-self/finite OK | test linked {np.mean(t0 >= 0):.4f} (own {np.mean(o0 >= 0):.4f}), OOF linked {np.mean(oo >= 0):.4f} "
          f"| test changed vs own (8 matchings) {np.round(ch, 3).tolist()} | OOF changed {och:.4f} | OOF exact {np.mean(oo[h] == ts[h]):.4f} (own {np.mean(own[fit]['oof_succ'][0][h] == ts[h]):.4f})")
    print(f"   same-limb share test new {same_new:.4f} own {same_own:.4f} | OOF new {np.mean(sens[oo[oo >= 0]] == sens[oo >= 0]):.4f} true {np.mean(sens[ts[h]] == sens[h]):.4f}")
    print(f"   chain agreement same-limb anchored links: TEST new {ca_new[0]:.4f} (n {ca_new[1]}, by limb {ca_new[2]}) own {ca_own[0]:.4f} | "
          f"OOF-sim new {ca_oof[0]:.4f} (n {ca_oof[1]}, by limb {ca_oof[2]}) own {ca_oof_own[0]:.4f}")
    print(f"   test score q10/50/90 {np.quantile(zf['test_score'][0][t0 >= 0], [.1, .5, .9]).round(3).tolist()} oof {np.quantile(zf['oof_score'][0][oo >= 0], [.1, .5, .9]).round(3).tolist()}")
    out[tag] = dict(test_changed=ch[0], chain_agree_test=ca_new[0], chain_agree_oof=ca_oof[0])
# K9 spec file OOF vs this run's own nested OOF
a_ = np.load(os.path.join(TD, "links_test_K9.npz")); b_ = np.load(os.path.join(TD, "links_oof_K9_spec.npz"))
print("C K9 spec file OOF vs links_oof_K9_spec: succ identical", bool((a_["oof_succ"] == b_["oof_succ"]).all()), "score max|diff|", float(np.abs(a_["oof_score"] - b_["oof_score"]).max()))
qa = own["K7"]["qn_ref"]; qb = own["K9"]["qn_ref"]
print("C qn_ref K7 vs K9: len", len(qa), len(qb), "quantiles K7", np.quantile(qa, [.1, .5, .9]).round(3).tolist(), "K9", np.quantile(qb, [.1, .5, .9]).round(3).tolist())
print("C own K9 test score quantiles", np.quantile(own["K9"]["test_score"][0][own["K9"]["test_succ"][0] >= 0], [.1, .5, .9]).round(3).tolist())
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit1.json"), "w"), indent=1)
print("DONE")
