"""Forks of goodpjw2008's public notebook "WEAR@HASCA | Learned Links + Counts | LB 0.92942" (version 4; built on
Briano / Hanh Tran / Jiwei Liu, Apache 2.0) as independent fits for ensembling. Changes (marked FORK):
  * WEAR_SEED set per fork (window models +10*seed, two-tower seeds seed and seed+1), LightGBM stage seeds shifted;
  * the public-pipeline OOF ablation is skipped (saves ~10 min);
  * every stage input of the decode is kept as output (keep4/), so that multi-fit decodes need no GPU:
    blends, tabular experts, bagged L3 links, link log-odds + candidate lists, count-regressor cache, embeddings,
    refiner scalars, final probabilities and refiner labels.
  python patch_v4.py <seed> [<seed> ...]   -> kaggle/v4_fork/s<seed>/{v4_s<seed>.ipynb, kernel-metadata.json}"""
import json, os, sys
W = r"E:\Claude code\wear"
SRC = os.path.join(W, "kaggle", "pub_1004", "good", "wear-hasca-learned-links-counts-lb-0-92942.ipynb")
HERE = os.path.join(W, "kaggle", "v4_fork")

DEC_OLD = "        ct = fit_counts(X, true, Xt)\n"
DEC_NEW = ("        ct = fit_counts(X, true, Xt)\n"
           "        DEC_CACHE.setdefault(tag, []).append((X.copy(), true.copy(), Xt.copy(), kt.copy()))   # FORK\n")
LO_OLD = "    del LO_OOF, LO_TEST"
LO_NEW = r'''    # FORK: keep the link log-odds and candidate lists of every subject (re-bagging / cross-fit link averaging offline)
    _KEEP = WORK / "keep4"; _KEEP.mkdir(exist_ok=True)
    _lo = {}
    for _s, _L in LO_OOF.items():
        _z = np.load(ROOT / f"pairs_{_s}.npz"); _lo[f"oof_{_s}_L"] = _L.astype(np.float32); _lo[f"oof_{_s}_cand"] = _z["cand"]
    for _s, _L in LO_TEST.items():
        _z = np.load(ROOT / f"testpairs_{_s}.npz"); _lo[f"test_{_s}_L"] = _L.astype(np.float32); _lo[f"test_{_s}_cand"] = _z["cand"]
    np.savez_compressed(_KEEP / "link_logodds.npz", **_lo)
    del _lo
    del LO_OOF, LO_TEST'''
FINAL = r'''
# ---- FORK: keep every stage input of the decode, so that multi-fit decodes can run offline
KEEP = WORK / "keep4"; KEEP.mkdir(exist_ok=True)
np.savez(KEEP / "stage.npz", B2_TEST=B2_TEST, B2_OOF=B2_OOF, TEST_LOGP_B=TEST_LOGP_B, OOF_LOGP_B=OOF_LOGP_B,
         QA_TEST=QA_TEST, QA_OOF=QA_OOF, QB_TEST=QB_TEST, QB_OOF=QB_OOF, PB_TEST=PB_TEST, PB_OOF=PB_OOF,
         TA_TEST=TA_TEST, TA_OOF=TA_OOF, test_logp=t["logp"], oof_logp=d["logp"],
         tab_S3_test=TAB_TEST["S3"], tab_T_test=TAB_TEST["T"], tab_S3_oof=TAB_OOF["S3"], tab_T_oof=TAB_OOF["T"],
         test_sbj=t["sbj"], oof_sbj=d["sbj"], oof_y=d["y"], oof_fold=d["fold"], ids=np.asarray(ids).astype(str),
         ref_oof=REF_OOF, ref_test=labels, sensor_oof=d["sensor"], sensor_test=t["sensor"], true_succ=TRUE_SUCC,
         oof_rec=sim["rec"], oof_start=sim["start"])
np.savez(KEEP / "links.npz", test_succ=np.stack([m[0] for m in L3_TEST_LINKS]), test_score=np.stack([m[1] for m in L3_TEST_LINKS]),
         oof_succ=np.stack([m[0] for m in L3_OOF_LINKS]), oof_score=np.stack([m[1] for m in L3_OOF_LINKS]),
         l2_test_succ=L2_SUCC, l2_test_score=L2_SCORE_QN, qn_ref=QN_REF)
_dc = {}
for _tg, _lst in DEC_CACHE.items():
    for _k, (_X, _tr, _Xt, _kt) in enumerate(_lst):
        _p = f"{_tg.replace(' ', '_')}_{_k}"
        _dc[_p + "_X"], _dc[_p + "_true"], _dc[_p + "_Xt"], _dc[_p + "_kt"] = _X, _tr, _Xt, _kt
np.savez(KEEP / "dec_cache.npz", **_dc)
np.save(KEEP / "oof_emb.npy", np.asarray(d["emb"]).astype(np.float16))
np.save(KEEP / "test_emb.npy", np.asarray(t["emb"]).astype(np.float16))
_so, _st = tile_scalars(d), tile_scalars(t)
np.savez(KEEP / "tile_scalars.npz", oof_ener=_so[0], oof_post=_so[1], oof_vmot=_so[2], oof_vmean=_so[3].astype(np.float16),
         test_ener=_st[0], test_post=_st[1], test_vmot=_st[2], test_vmean=_st[3].astype(np.float16))
print("FORK kept:", sorted(p.name for p in KEEP.iterdir()), flush=True)
'''


def patch(seed, extra=None, name=None):
    """extra: optional list of (old, new, expected_count) replacements applied to the code cells after the base patch"""
    nb = json.load(open(SRC, encoding="utf-8"))
    xn = [0] * len(extra or [])
    n = dict(env=0, abl=0, dec=0, cache=0, lo=0, fin=0, lgb=0)
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if 'ABLATION = os.environ.get("WEAR_ABLATION", "1") == "1"' in s:
            s = s.replace('ABLATION = os.environ.get("WEAR_ABLATION", "1") == "1"',
                          f'os.environ["WEAR_SEED"] = "{seed}"   # FORK: independent fit\n'
                          'ABLATION = False   # FORK: skip the OOF ablations of the intermediate steps')
            n["env"] += 1; n["abl"] += 1
        if "def decode(B_oof, B_test, tag, passes=2):" in s:
            s = s.replace("def decode(B_oof, B_test, tag, passes=2):",
                          "DEC_CACHE = {}   # FORK: profile features per stage and pass\n\n\ndef decode(B_oof, B_test, tag, passes=2):")
            n["dec"] += 1
            assert DEC_OLD in s
            s = s.replace(DEC_OLD, DEC_NEW); n["cache"] += 1
        if LO_OLD in s:
            s = s.replace(LO_OLD, LO_NEW); n["lo"] += 1
        for a, b in (("verbose=-1, seed=0)", f"verbose=-1, seed={seed})"), ("verbosity=-1, seed=0)", f"verbosity=-1, seed={seed})")):
            if a in s:
                s = s.replace(a, b); n["lgb"] += 1
        if "if CLEANUP and ROOT.exists():" in s:
            s = s.replace("if CLEANUP and ROOT.exists():", FINAL + "\nif CLEANUP and ROOT.exists():"); n["fin"] += 1
        for i, (old, new, _) in enumerate(extra or []):
            if old in s:
                xn[i] += s.count(old); s = s.replace(old, new)
        c["source"] = s.splitlines(keepends=True)
    assert all(n[k] == 1 for k in ("env", "abl", "dec", "cache", "lo", "fin")) and n["lgb"] == 2, n
    # with ABLATION off, decode() does not write the three ablation-only scores that stage O4 reads: read them with
    # .get(), otherwise every fork dies with a KeyError at O4 (found by review, 2026-10-04)
    src_all = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    nk = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        for key in ("one matching, public configs, fixed 97", "bagged, public configs, fixed 97", "bagged, g 1.5, raw embedding, fixed 97"):
            old = f'= _r["{key}"]'
            if old in s:
                s = s.replace(old, f'= _r.get("{key}", float("nan"))   # FORK'); nk += 1
        c["source"] = s.splitlines(keepends=True)
    assert nk == 3, nk
    assert not any('_r["one matching, public configs' in "".join(c["source"]) for c in nb["cells"]), "unguarded lookup left"
    del src_all
    for i, (old, _, cnt) in enumerate(extra or []):
        assert xn[i] == cnt, (old[:80], xn[i], cnt)
    name = name or f"v4-s{seed}"; fn = name.replace("-", "_")
    out = os.path.join(HERE, name.replace("v4-", "")); os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, f"{fn}.ipynb"), "w", encoding="utf-8"))
    json.dump({"id": f"koushikrudra/wear-{name}", "title": f"wear-{name}", "code_file": f"{fn}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
               "dataset_sources": [], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"], "kernel_sources": [],
               "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    print(name, "seed", seed, "patched", n, "extra", xn, "->", out)
    return out


if __name__ == "__main__":
    for sd in sys.argv[1:]:
        patch(int(sd))
