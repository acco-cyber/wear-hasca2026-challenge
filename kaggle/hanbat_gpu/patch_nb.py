"""Fork of woominyo's public notebook (Apache 2.0) for a GPU T4x2 run that also keeps the intermediate arrays
(window-model OOF/test log-probs, embeddings, links, tabular features/experts, graph output) as notebook output."""
import json, os
SRC = r"E:\Claude code\wear\public_src\woominyo\lb-0-890-wear-hasca-2026-timeline-graph.ipynb"
DST = r"E:\Claude code\wear\kaggle\hanbat_gpu\hanbat_gpu.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))

FEAT_KEEP = '''
# --- keep tabular features for offline re-runs (fork addition)
KEEP = WORK / "keep"; KEEP.mkdir(exist_ok=True)
np.savez(KEEP / "feat_oof.npz", **FEAT["oof"]); np.savez(KEEP / "feat_test.npz", **FEAT["test"])
json.dump(IMU_NAMES, open(KEEP / "imu_names.json", "w"))
print("kept tabular features")
'''

FINAL_KEEP = '''
# --- keep intermediate arrays (fork addition)
KEEP = WORK / "keep"; KEEP.mkdir(exist_ok=True)
import shutil as _sh
for r in FUSION + IMU:
    o = np.load(r / "oof_raw.npz")
    np.savez(KEEP / f"oof_{r.name}.npz", logp=o["logp"], y=o["y"], grp=o["grp"])
    _sh.copy(r / "test_logp_raw.npy", KEEP / f"test_logp_raw_{r.name}.npy")
    if (r / "log.txt").exists():
        _sh.copy(r / "log.txt", KEEP / f"log_{r.name}.txt")
np.save(KEEP / "oof_emb.npy", d["emb"].astype(np.float16))
np.save(KEEP / "test_emb.npy", t["emb"].astype(np.float16))
for f in ("sim_meta.npz", "links_L0.npz", "test_logp_b.npy", "links_L2_test.npz", "P_test.npy"):
    if (ROOT / f).exists():
        _sh.copy(ROOT / f, KEEP / f)
np.savez(KEEP / "blend.npz", oof_logp=d["logp"], oof_fusion=d["fusion_logp"], oof_imu=d["imu_logp"],
         test_logp=t["logp"], test_fusion=t["fusion_logp"], test_imu=t["imu_logp"], plp_oof=PLP_OOF, plp_test=PLP_TEST,
         tab_S3=TAB_TEST["S3"], tab_T=TAB_TEST["T"], test_sensor=t["sensor"], test_sbj=t["sbj"], labels=labels)
_sh.copy(PROC / "test_id.npy", KEEP / "test_id.npy")
np.save(KEEP / "sim_acc.npy", np.load(ROOT / "sim_acc.npy"))
print("kept:", sorted(p.name for p in KEEP.iterdir()))
'''

n_feat = n_final = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if 'with stage("tabular features"):' in src:
        src = src.rstrip("\n") + "\n" + FEAT_KEEP
        n_feat += 1
    if "if CLEANUP and ROOT.exists():" in src:
        src = src.replace("if CLEANUP and ROOT.exists():", FINAL_KEEP + "\nif CLEANUP and ROOT.exists():")
        n_final += 1
    c["source"] = src.splitlines(keepends=True)
assert n_feat == 1 and n_final == 1, (n_feat, n_final)
# title
for c in nb["cells"]:
    if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("# WEAR@HASCA 2026 | Timeline Reconstruction"):
        lines = "".join(c["source"]).splitlines(keepends=True)
        c["source"] = ["# Fork (GPU T4x2 run, keeps intermediates) of woominyo's \"WEAR@HASCA 2026 | Timeline Reconstruction + Graph\" (Apache 2.0)\n", "\n"] + lines[1:]
        break
nb.setdefault("metadata", {})
json.dump(nb, open(DST, "w", encoding="utf-8"))
meta = {"id": "koushikrudra/wear-hanbat-gpu", "title": "wear-hanbat-gpu", "code_file": "hanbat_gpu.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"],
        "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open(os.path.join(os.path.dirname(DST), "kernel-metadata.json"), "w"), indent=1)
print("patched", DST, "cells", len(nb["cells"]))
