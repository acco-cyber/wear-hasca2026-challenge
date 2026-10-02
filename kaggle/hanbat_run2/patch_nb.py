"""Fork of jiweiliu's 'public fast GPU inference' (Apache 2.0, derived from woominyo's Timeline + Graph notebook): a second,
independently fitted copy of the pipeline (models from the public dataset jiweiliu/wear-timeline-graph-models).
The fork only adds a cell part that keeps the test intermediates as output (keep2/)."""
import json, os
SRC = r"E:\Claude code\wear\public_src\jiweiliu\public-fast-gpu-inference.ipynb"
DST = r"E:\Claude code\wear\kaggle\hanbat_run2\hanbat_run2.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))
KEEP = '''
# --- keep test intermediates (fork addition)
KEEP = WORK / "keep2"; KEEP.mkdir(exist_ok=True)
np.save(KEEP / "test_logp_b.npy", TEST_LOGP_B); np.save(KEEP / "P_test.npy", P_TEST)
np.savez(KEEP / "links_L2_test.npz", succ=L2_SUCC, score=L2_SCORE, score_qn=L2_SCORE_QN)
np.savez(KEEP / "blend.npz", test_logp=t["logp"], tab_S3=TAB_TEST["S3"], tab_T=TAB_TEST["T"], labels=labels, test_sbj=t["sbj"])
np.save(KEEP / "test_emb.npy", t["emb"].astype(np.float16))
print("kept:", sorted(p.name for p in KEEP.iterdir()))
'''
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "if CLEANUP and ROOT.exists():" in src:
        src = src.replace("if CLEANUP and ROOT.exists():", KEEP + "\nif CLEANUP and ROOT.exists():"); n += 1
    c["source"] = src.splitlines(keepends=True)
assert n == 1, n
json.dump(nb, open(DST, "w", encoding="utf-8"))
meta = {"id": "koushikrudra/wear-hanbat-run2", "title": "wear-hanbat-run2", "code_file": "hanbat_run2.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["jiweiliu/wear-timeline-graph-models"], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"],
        "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open(os.path.join(os.path.dirname(DST), "kernel-metadata.json"), "w"), indent=1)
print("patched", DST)
