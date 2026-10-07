import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for F in ("K7", "K9"):
    I = json.load(open(os.path.join(HERE, f"info_{F}.json")))
    ss = I["sim_per_subject"]
    for k in ("coverage_ge10", "member0_changed", "asm_link_share"):
        v = np.array([ss[s][k] for s in ss])
        t = [round(I["test_per_subject"][s][k], 4) for s in sorted(I["test_per_subject"])]
        print(F, k, "sim per-subject min/median/max", np.round([v.min(), np.median(v), v.max()], 4), "| test 22-25", t)
