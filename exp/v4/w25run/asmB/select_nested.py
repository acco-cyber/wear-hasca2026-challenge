"""asmB: nested choice of the one tuned threshold (wmin) by subject fold: for fold f pick the wmin whose member-0 exact
successor rate is best on the subjects of the other folds, take fold f's links from that run. Writes the 8-member OOF
links + K7 test placeholder (links_test_K7_deaug.npz test arrays) -> links_asmB_nested.npz"""
import os, json
import numpy as np
from asmlib import stage, TD, HERE

S = stage(); fold = S["oof_fold"]; ts = S["true_succ"]; h = ts >= 0; y = S["oof_y"]
CANDS = ["0.3", "0.5", "0.8"]
Z = {c: np.load(os.path.join(HERE, f"oof_links_w{c}.npz")) for c in CANDS}
base = np.load(os.path.join(TD, "links_test_K7_deaug.npz"))
ex = {c: (Z[c]["oof_succ"][0] == ts) for c in CANDS}
ex["base"] = base["oof_succ"][0] == ts
Su = np.full_like(base["oof_succ"], -1); Sc = np.full_like(base["oof_score"], -50.0); pick = {}
for f in range(5):
    tr = h & (fold != f)
    sc = {c: float(ex[c][tr].mean()) for c in CANDS}
    c = max(CANDS, key=lambda k: sc[k]); pick[f] = c
    m = fold == f; Su[:, m] = Z[c]["oof_succ"][:, m]; Sc[:, m] = Z[c]["oof_score"][:, m]
    print(f"fold {f}: train-fold exact {json.dumps({k: round(v, 4) for k, v in sc.items()})} -> wmin {c}; "
          f"fold exact {ex[c][h & m].mean():.4f} vs base {ex['base'][h & m].mean():.4f}")
e = Su[0][h] == ts[h]; eb = ex["base"][h]
lk = Su[0] >= 0; lkb = base["oof_succ"][0] >= 0
xw = lambda su: float(np.mean(np.where(su >= 0, (y != y[np.maximum(su, 0)]) & (su != ts), False)[su >= 0]))
print(f"NESTED: exact {e.mean():.4f} vs base {eb.mean():.4f}; wrong cross-label share {xw(Su[0]):.4f} vs base {xw(base['oof_succ'][0]):.4f}; "
      f"linked {lk.mean():.4f} vs {lkb.mean():.4f}; picks {pick}")
# one-to-one check per member
for k in range(Su.shape[0]):
    v = Su[k][Su[k] >= 0]; assert len(np.unique(v)) == len(v), k
    assert (Su[k] != np.arange(len(Su[k]))).all()
np.savez(os.path.join(HERE, "links_asmB_nested.npz"), oof_succ=Su, oof_score=Sc, test_succ=base["test_succ"], test_score=base["test_score"])
json.dump(dict(picks=pick, exact=float(e.mean()), exact_base=float(eb.mean())), open(os.path.join(HERE, "nested_info.json"), "w"), indent=1)
print("wrote links_asmB_nested.npz")
