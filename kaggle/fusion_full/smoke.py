"""Local CPU smoke test of fusion_full.py (1 seed, 1 epoch, 5 sessions, PCA-160 prep data)."""
import os
os.environ["WEAR_LOCAL"] = "1"; os.environ["WEAR_SMOKE"] = "1"; os.environ["WEAR_EPOCHS"] = "1"
src = open(r"E:\Claude code\wear\kaggle\fusion_full\fusion_full.py", encoding="utf-8").read().replace("SEEDS = [0, 1, 2]", "SEEDS = [0]")
exec(compile(src, "fusion_full_smoke", "exec"))
import numpy as np
p = np.load(r"E:\Claude code\wear\work\fusion_local\test.npy"); print("SMOKE OK test.npy", p.shape, round(float(p.sum(1).mean()), 4))
