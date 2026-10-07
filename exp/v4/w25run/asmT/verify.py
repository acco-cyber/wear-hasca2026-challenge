"""asmT: check links_asm_K7 reproduces asmB (nested OOF + real test) and summarise both written files."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(os.path.dirname(HERE), "asmB"); TD = os.path.join(os.path.dirname(HERE), "testD")
t7 = np.load(os.path.join(HERE, "links_asm_K7.npz"))
bn = np.load(os.path.join(B, "links_asmB_nested.npz")); bt = np.load(os.path.join(B, "links_asmB_test.npz"))
print("K7 oof_succ == asmB nested:", (t7["oof_succ"] == bn["oof_succ"]).all(), "score max|d|", float(np.abs(t7["oof_score"] - bn["oof_score"]).max()))
print("K7 test_succ == asmB test:", (t7["test_succ"] == bt["test_succ"]).all(), "score max|d|", float(np.abs(t7["test_score"] - bt["test_score"]).max()))
for F in ("K7", "K9"):
    p = os.path.join(HERE, f"links_asm_{F}.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p); base = np.load(os.path.join(TD, f"links_test_{F}_deaug.npz"))
    print(F, {k: (z[k].shape, str(z[k].dtype)) for k in z.files},
          "test m0 changed", round(float(np.mean(z["test_succ"][0] != base["test_succ"][0])), 4),
          "oof m0 changed", round(float(np.mean(z["oof_succ"][0] != base["oof_succ"][0])), 4),
          "test linked", round(float(np.mean(z["test_succ"][0] >= 0)), 4), "vs", round(float(np.mean(base["test_succ"][0] >= 0)), 4))
if os.path.exists(os.path.join(HERE, "links_asm_K9.npz")):
    z7 = np.load(os.path.join(HERE, "links_asm_K7.npz")); z9 = np.load(os.path.join(HERE, "links_asm_K9.npz"))
    print("K7 vs K9 asm test member0 agreement", round(float(np.mean(z7["test_succ"][0] == z9["test_succ"][0])), 4))
