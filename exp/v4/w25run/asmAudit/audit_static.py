"""asmAudit: static checks of the asmT link files (read-only on everything outside this folder).
Format, one-to-one members, links inside subjects, equality with asmB, K9 stage == K7 stage (load_sim uses K7 stage),
index conventions (OOF tile order vs time, L3 candidate indexing, test cache indexing)."""
import os, sys, json
import numpy as np
W = r"E:\Claude code\wear"; R = os.path.join(W, "exp", "v4", "w25run")
T = os.path.join(R, "asmT"); B = os.path.join(R, "asmB"); TD = os.path.join(R, "testD"); CACHE = os.path.join(TD, "cache")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4"); K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
FITS = {"K7": K7, "K9": K9}
out = {}
st = {F: np.load(os.path.join(p, "stage.npz"), allow_pickle=True) for F, p in FITS.items()}
keys = ["oof_y", "oof_sbj", "oof_fold", "oof_rec", "oof_start", "true_succ", "sensor_oof", "test_sbj", "sensor_test"]
out["k9_stage_equals_k7"] = {k: bool(np.array_equal(st["K7"][k], st["K9"][k])) for k in keys}
S = {k: st["K7"][k].astype(np.int64) for k in keys}
sbj, tsbj, ts = S["oof_sbj"], S["test_sbj"], S["true_succ"]; N, NT = len(sbj), len(tsbj)
out["test_subjects"] = {int(s): int((tsbj == s).sum()) for s in np.unique(tsbj)}

# ---- OOF tile order vs time: share of subject-local consecutive pairs (i, i+1) that are true successors
cons = []
for s in np.unique(sbj):
    loc = np.flatnonzero(sbj == s); cons.append(ts[loc[:-1]] == loc[1:])
out["oof_local_order_is_time_share"] = float(np.mean(np.concatenate(cons)))

# ---- link files
def chk(Su, grp, nm):
    r = {}
    r["shape"] = list(Su.shape)
    r["one_to_one"] = all(len(np.unique(Su[k][Su[k] >= 0])) == int((Su[k] >= 0).sum()) for k in range(Su.shape[0]))
    r["no_self"] = bool((Su != np.arange(Su.shape[1])[None]).all())
    r["in_subject"] = bool(np.where(Su >= 0, grp[np.maximum(Su, 0)] == grp[None], True).all())
    r["linked_m0"] = float(np.mean(Su[0] >= 0))
    return r

for F in ("K7", "K9"):
    z = np.load(os.path.join(T, f"links_asm_{F}.npz")); base = np.load(os.path.join(TD, f"links_test_{F}_deaug.npz"))
    oofb = np.load(os.path.join(TD, f"links_oof_{F}_deaug.npz"))
    r = {"files": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
    r["oof"] = chk(z["oof_succ"], sbj, "oof"); r["test"] = chk(z["test_succ"], tsbj, "test")
    r["base_test"] = chk(base["test_succ"], tsbj, "bt"); r["base_oof"] = chk(oofb["oof_succ"], sbj, "bo")
    r["test_scores_finite"] = bool(np.isfinite(z["test_score"]).all()); r["oof_scores_finite"] = bool(np.isfinite(z["oof_score"]).all())
    hi = float(np.percentile(np.load(os.path.join(FITS[F], "links.npz"))["qn_ref"], 99)); r["high"] = hi
    r["test_score_pct_linked"] = np.percentile(z["test_score"][z["test_succ"] >= 0], [1, 50, 90, 99, 100]).round(3).tolist()
    r["base_test_score_pct_linked"] = np.percentile(base["test_score"][base["test_succ"] >= 0], [1, 50, 90, 99, 100]).round(3).tolist()
    r["test_share_high"] = float(np.mean(np.isclose(z["test_score"][0], hi)))
    r["unlinked_score_values"] = np.unique(z["test_score"][z["test_succ"] < 0]).round(3)[:5].tolist()
    r["test_m0_changed"] = float(np.mean(z["test_succ"][0] != base["test_succ"][0]))
    r["oof_m0_changed"] = float(np.mean(z["oof_succ"][0] != oofb["oof_succ"][0]))
    h = ts >= 0
    r["oof_exact_m0"] = float(np.mean(z["oof_succ"][0][h] == ts[h])); r["oof_exact_m0_base"] = float(np.mean(oofb["oof_succ"][0][h] == ts[h]))
    # test: share of member-0 links pointing to the next pipeline index (would be ~0 if test order is shuffled)
    r["test_m0_to_next_index"] = float(np.mean(z["test_succ"][0] == np.arange(NT) + 1))
    r["base_test_m0_to_next_index"] = float(np.mean(base["test_succ"][0] == np.arange(NT) + 1))
    # L3 candidates: subject-local indexing check
    LZ = np.load(os.path.join(FITS[F], "link_logodds.npz")); l3o, l3t = [], []
    for s in np.unique(sbj):
        loc = np.flatnonzero(sbj == s); l_of = np.full(N, -1); l_of[loc] = np.arange(len(loc))
        c = LZ[f"oof_{s}_cand"]; L = LZ[f"oof_{s}_L"]; top = c[np.arange(len(c)), np.argmax(np.where(c >= 0, L, -1e9), 1)]
        tl = np.where(ts[loc] >= 0, l_of[np.maximum(ts[loc], 0)], -1); m = tl >= 0; l3o.append(top[m] == tl[m])
        assert c.shape[0] == len(loc)
    for s in (22, 23, 24, 25):
        loc = np.flatnonzero(tsbj == s); l_of = np.full(NT, -1); l_of[loc] = np.arange(len(loc))
        c = LZ[f"test_{s}_cand"]; L = LZ[f"test_{s}_L"]; top = c[np.arange(len(c)), np.argmax(np.where(c >= 0, L, -1e9), 1)]
        b0 = base["test_succ"][0][loc]; b0l = np.where(b0 >= 0, l_of[np.maximum(b0, 0)], -1); m = b0l >= 0; l3t.append(top[m] == b0l[m])
        assert c.shape[0] == len(loc)
    r["l3_top1_exact_oof"] = float(np.mean(np.concatenate(l3o))); r["l3_top1_agree_base_m0_test"] = float(np.mean(np.concatenate(l3t)))
    out[F] = r

# ---- equality with asmB
t7 = np.load(os.path.join(T, "links_asm_K7.npz")); bn = np.load(os.path.join(B, "links_asmB_nested.npz")); bt = np.load(os.path.join(B, "links_asmB_test.npz"))
out["K7_eq_asmB"] = dict(oof=bool((t7["oof_succ"] == bn["oof_succ"]).all()), test=bool((t7["test_succ"] == bt["test_succ"]).all()),
                         oof_score_maxdiff=float(np.abs(t7["oof_score"] - bn["oof_score"]).max()),
                         test_score_maxdiff=float(np.abs(t7["test_score"] - bt["test_score"]).max()))
# ---- test chain cache indexing
tc = {}
for s in (22, 23, 24, 25):
    cf = np.load(os.path.join(CACHE, f"testdeaug_chain_s{s}.npz")); loc = np.flatnonzero(tsbj == s); n = len(loc)
    anch = cf["anch"].astype(bool); pos = cf["pos"].astype(np.int64); lim = cf["lim"].astype(np.int64); ow = cf["owner"]
    okpos = bool(((pos[anch] >= 0) & (pos[anch] < n)).all())
    back = bool((ow[lim[anch], pos[anch]] == np.flatnonzero(anch)).all())
    nown = {L: int((ow[L] >= 0).sum()) for L in range(4)}; nan = {L: int((anch & (lim == L)).sum()) for L in range(4)}
    succ_ok = all(len(cf[f"succ{L}"]) == n and len(cf[f"rows{L}"]) == n for L in range(4))
    one = all(len(np.unique(cf[f"succ{L}"][cf[f"succ{L}"] >= 0])) == int((cf[f"succ{L}"] >= 0).sum()) for L in range(4))
    tc[s] = dict(n=n, loc_eq=bool((cf["loc"] == loc).all()), lim_eq=bool((lim == S["sensor_test"][loc]).all()), pos_in_range=okpos,
                 owner_back=back, owner_count_eq_anch=nown == nan, chains_len_n=succ_ok, chains_one_to_one=one,
                 anch_rate_by_limb={L: round(float(anch[lim == L].mean()), 4) for L in range(4)})
out["test_cache"] = tc
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_static.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
