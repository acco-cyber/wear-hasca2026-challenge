"""Second fork of the same notebook (see kaggle/good_fork/patch_nb.py) with different random seeds for the window
models, the two-tower matcher and the LightGBM stages: an independent fit for ensembling the final probabilities."""
import json, os
W = r"E:\Claude code\wear"; SRC = os.path.join(W, "kaggle", "good_fork", "good_fork.ipynb"); HERE = os.path.join(W, "kaggle", "good_fork2")
nb = json.load(open(SRC, encoding="utf-8")); n = dict(runs=0, tt=0, lgb=0)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if '("final_s0", "--seed 0 --epochs 12' in s:
        for a, b in (("--seed 0 --epochs 12", "--seed 10 --epochs 12"), ("--seed 0 --epochs 30", "--seed 10 --epochs 30"), ("--seed 1 --epochs 12", "--seed 11 --epochs 12"),
                     ("--seed 2 --epochs 12", "--seed 12 --epochs 12"), ("--seed 3 --epochs 12", "--seed 13 --epochs 12"), ("--seed 1 --epochs 30", "--seed 11 --epochs 30")):
            assert a in s, a; s = s.replace(a, b)
        n["runs"] += 1
    if 'p.add_argument("--seed", type=int, default=0)' in s:
        s = s.replace('p.add_argument("--seed", type=int, default=0)', 'p.add_argument("--seed", type=int, default=7)'); n["tt"] += 1
    for a, b in (("verbose=-1, seed=0)", "verbose=-1, seed=1)"), ("verbosity=-1, seed=0)", "verbosity=-1, seed=1)")):
        if a in s:
            s = s.replace(a, b); n["lgb"] += 1
    c["source"] = s.splitlines(keepends=True)
print(n); assert n["runs"] == 1 and n["tt"] >= 1 and n["lgb"] == 2
json.dump(nb, open(os.path.join(HERE, "good_fork2.ipynb"), "w", encoding="utf-8"))
json.dump({"id": "koushikrudra/wear-good-fork2", "title": "wear-good-fork2", "code_file": "good_fork2.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": ["koushikrudra/wear-extra-links"],
           "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"], "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
          open(os.path.join(HERE, "kernel-metadata.json"), "w"), indent=1)
