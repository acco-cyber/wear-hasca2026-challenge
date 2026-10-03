"""Fork of goodpjw2008's public notebook "WEAR@HASCA | Learned Links + Counts" (built on Briano / Hanh Tran / Jiwei Liu,
Apache 2.0). Additions (marked FORK): extra independent one-to-one link sets (our chain-link scorer, two fits of the public
L2 links) can enter the graph stage as additional edges or as extra bag members; several test-side variants are decoded;
all stage inputs are kept as output so that further variants need no GPU."""
import json, os, pickle, shutil
import numpy as np
W = r"E:\Claude code\wear"; HERE = os.path.join(W, "kaggle", "good_fork"); DS = os.path.join(W, "kaggle", "extra_links")
SRC = os.path.join(W, "public_src", "good927", "wear-hasca-learned-links-counts-lb-0-92718.ipynb")
os.makedirs(DS, exist_ok=True)
# ---- dataset with our link sets (test rows, global successor index, score)
st = pickle.load(open(os.path.join(W, "work", "test_structure.pkl"), "rb")); N = 12234
succ = np.full(N, -1, np.int64); sc = np.full(N, -50.0, np.float32)
for s, d in st.items():
    idx = np.asarray(d["idx"]); su = np.asarray(d["succ0"]); c = np.asarray(d["sc"], np.float32); ok = (su >= 0) & (c >= -6.0)
    succ[idx[ok]] = idx[su[ok]]; sc[idx[ok]] = c[ok]
np.savez(os.path.join(DS, "links_our_test.npz"), succ=succ, score_qn=sc)          # raw log-odds, used with bias -2 as in gl7
shutil.copy(os.path.join(W, "work", "hanbat", "links_ourfull_test.npz"), os.path.join(DS, "links_ourfull_test.npz"))
shutil.copy(os.path.join(W, "work", "hanbat", "keep", "links_L2_test.npz"), os.path.join(DS, "links_hb_run1_L2.npz"))
shutil.copy(os.path.join(W, "work", "hanbat", "run2", "keep2", "links_L2_test.npz"), os.path.join(DS, "links_hb_run2_L2.npz"))
json.dump({"title": "wear-extra-links", "id": "koushikrudra/wear-extra-links", "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(DS, "dataset-metadata.json"), "w"), indent=1)

nb = json.load(open(SRC, encoding="utf-8"))
RUNNER_OLD = '''        self.Lw = link_W(dd["succ"], dd["score"], len(self.P0), b=b)
        self.knns, self.K, self.stage1 = {}, K, {}'''
RUNNER_NEW = '''        self.Lw = link_W(dd["succ"], dd["score"], len(self.P0), b=b)
        for _su, _sc, _w, _b in dd.get("xlinks", []):       # FORK: extra one-to-one link sets as additional edges
            self.Lw = self.Lw + _w * link_W(_su, _sc, len(self.P0), b=_b)
        self.knns, self.K, self.stage1 = {}, K, {}'''
FINAL = r'''
# ---- FORK: test-side variants with extra independent link sets, and keep the stage inputs
_xl = next(Path("/kaggle/input").rglob("links_our_test.npz")).parent
def _ld(name):
    z = np.load(_xl / name); return z["succ"].astype(np.int64), z["score_qn"].astype(np.float32)
OUR, OURF, HB1, HB2 = _ld("links_our_test.npz"), _ld("links_ourfull_test.npz"), _ld("links_hb_run1_L2.npz"), _ld("links_hb_run2_L2.npz")


def decode_test(B_test, base_tag, links, xlinks=None):
    dt = dict(logp=B_test, emb=t["emb"], grp=t["sbj"], sbj=t["sbj"], sets={})
    if xlinks:
        dt["xlinks"] = xlinks
    Pt = bag_P(dt, links)[0]; Btp = np.exp(B_test.astype(np.float64)); n = len(DEC_CACHE[base_tag])
    for k, (X, true) in enumerate(DEC_CACHE[base_tag]):
        Xt, kt = profile_features([Pt, Btp], t["sbj"], {}); ct = fit_counts(X, true, Xt); tgt = count_targets(t["sbj"], {}, kt, ct)
        if k < n - 1:
            Pt = bag_P(dt, links, tgt)[0]
    return finish_targets(Pt, t["sbj"], tgt), Pt, tgt


VARIANTS = {
    "v0_plain": (L3_TEST_LINKS, None),
    "v1_union_our": (L3_TEST_LINKS, [(OUR[0], OUR[1], 1.0, -2.0)]),
    "v2_union_our_hb2": (L3_TEST_LINKS, [(OUR[0], OUR[1], 1.0, -2.0), (HB2[0], HB2[1], 1.0, -2.0)]),
    "v3_union_our_hb1_hb2": (L3_TEST_LINKS, [(OUR[0], OUR[1], 1.0, -2.0), (HB1[0], HB1[1], 0.5, -2.0), (HB2[0], HB2[1], 0.5, -2.0)]),
    "v4_members_our_hb2": (L3_TEST_LINKS + [OUR, HB2], None),
    "v5_union_our05": (L3_TEST_LINKS, [(OUR[0], OUR[1], 0.5, -2.0)]),
}
KEEP = WORK / "keep3"; KEEP.mkdir(exist_ok=True)
with stage("FORK variants"):
    _out = {}
    for _nm, (_lk, _xk) in VARIANTS.items():
        _Q, _Pt, _tg = decode_test(B2_TEST, "stage B", _lk, _xk)
        _out[_nm + "_Q"] = _Q.astype(np.float32); _out[_nm + "_P"] = _Pt.astype(np.float32)
        print(f"{_nm}: agreement with the notebook's labels {(_Q.argmax(1) == labels).mean():.4f}", flush=True)
        pd.DataFrame({"id": ids, "target_feature": _Q.argmax(1).astype(int)}).sort_values("id").to_csv(WORK / f"submission_{_nm}.csv", index=False)
    np.savez(KEEP / "variants.npz", **_out)
np.savez(KEEP / "stage.npz", B2_TEST=B2_TEST, B2_OOF=B2_OOF, TEST_LOGP_B=TEST_LOGP_B, OOF_LOGP_B=OOF_LOGP_B, QA_TEST=QA_TEST, QA_OOF=QA_OOF,
         QB_TEST=QB_TEST, QB_OOF=QB_OOF, TA_TEST=TA_TEST, TA_OOF=TA_OOF, test_logp=t["logp"], oof_logp=d["logp"],
         tab_S3_test=TAB_TEST["S3"], tab_T_test=TAB_TEST["T"], tab_S3_oof=TAB_OOF["S3"], tab_T_oof=TAB_OOF["T"],
         test_sbj=t["sbj"], oof_sbj=d["sbj"], oof_y=d["y"], oof_fold=d["fold"], ids=ids)
np.savez(KEEP / "links.npz", test_succ=np.stack([m[0] for m in L3_TEST_LINKS]), test_score=np.stack([m[1] for m in L3_TEST_LINKS]),
         oof_succ=np.stack([m[0] for m in L3_OOF_LINKS]), oof_score=np.stack([m[1] for m in L3_OOF_LINKS]),
         l2_test_succ=L2_SUCC, l2_test_score=L2_SCORE_QN)
np.savez(KEEP / "dec_cache.npz", **{f"{tg.replace(' ', '_')}_{k}_{nm}": arr for tg, lst in DEC_CACHE.items() for k, (X, tr) in enumerate(lst) for nm, arr in (("X", X), ("true", tr))})
np.save(KEEP / "test_emb.npy", t["emb"].astype(np.float16))
print("kept:", sorted(p.name for p in KEEP.iterdir()))
'''
n = dict(abl=0, run=0, dec=0, cache=0, fin=0)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if 'ABLATION = os.environ.get("WEAR_ABLATION", "1") == "1"' in s:
        s = s.replace('ABLATION = os.environ.get("WEAR_ABLATION", "1") == "1"', "ABLATION = False   # FORK: skip the public-pipeline OOF ablation"); n["abl"] += 1
    if RUNNER_OLD in s:
        s = s.replace(RUNNER_OLD, RUNNER_NEW); n["run"] += 1
    if "def decode(B_oof, B_test, tag, passes=2):" in s:
        s = s.replace("def decode(B_oof, B_test, tag, passes=2):", "DEC_CACHE = {}   # FORK: OOF profile features per stage and pass\n\n\ndef decode(B_oof, B_test, tag, passes=2):"); n["dec"] += 1
        s = s.replace("        ct = fit_counts(X, true, Xt)\n", "        ct = fit_counts(X, true, Xt)\n        DEC_CACHE.setdefault(tag, []).append((X.copy(), true.copy()))\n"); n["cache"] += 1
    if "if CLEANUP and ROOT.exists():" in s:
        s = s.replace("if CLEANUP and ROOT.exists():", FINAL + "\nif CLEANUP and ROOT.exists():"); n["fin"] += 1
    c["source"] = s.splitlines(keepends=True)
assert all(v == 1 for v in n.values()), n
json.dump(nb, open(os.path.join(HERE, "good_fork.ipynb"), "w", encoding="utf-8"))
json.dump({"id": "koushikrudra/wear-good-fork", "title": "wear-good-fork", "code_file": "good_fork.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": ["koushikrudra/wear-extra-links"],
           "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"], "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
          open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=1)
print("patched", n)
