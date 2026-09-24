"""Assemble 5-fold OOF -> <tag>_cv/oof.npy, single-limb OOF macro-F1, blends into the e7 base for the sim check,
and test agreement with the akhyar public submission.  python assemble_eval.py [tag]"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\exp\base")
from common import eval_single, meta, to_log, norm_probs
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__)); tag = sys.argv[1] if len(sys.argv) > 1 else "aka"
cvd = os.path.join(HERE, f"{tag}_cv"); N = len(meta())
oof = np.full((N, 4, 19), np.nan, np.float32)
for k in range(5):
    r = np.load(os.path.join(cvd, f"rows_f{k}.npy")); oof[r] = np.load(os.path.join(cvd, f"oof_f{k}.npy"))
np.save(os.path.join(cvd, "oof.npy"), oof)
nanrow = np.isnan(oof[:, :, 0]); print("NaN limb-cells", nanrow.sum(), "rows fully NaN", nanrow.all(1).sum())
f = eval_single(oof, name=f"{tag} single-limb")
m = meta(); y = m.y.to_numpy(); ok = (m.pur.to_numpy() >= 0.8)
# per-limb macro-F1 (pur>=0.8)
for j, ln in enumerate(["left_arm", "left_leg", "right_arm", "right_leg"]):
    v = ok & ~nanrow[:, j]; print(f"  limb {ln}: macro-F1 {f1_score(y[v], oof[v, j].argmax(1), average='macro'):.4f}")
# compare with our base models
ref = {"v3b": r"E:\Claude code\wear\exp\base\v3b\oof.npy", "lgbm_v1": r"E:\Claude code\wear\work\lgbm_v1\oof.npy",
       "fusion": r"E:\Claude code\wear\work\fusion_v1\oof.npy", "base": r"E:\Claude code\wear\exp\base\bl_v3b_v1_f\oof.npy"}
R = {k: np.load(p).astype(np.float32) for k, p in ref.items()}
chk = norm_probs(0.5 * to_log(R["v3b"]) + 0.3 * to_log(R["lgbm_v1"]) + 0.2 * to_log(R["fusion"]))
d = np.nanmax(np.abs(chk - R["base"])); print(f"base == 0.5v3b+0.3v1+0.2f log-blend: max|diff| {d:.2e}")
both = ~np.isnan(R["base"][:, :, 0]) & ~nanrow
print(f"argmax agreement aka vs base (valid cells) {np.mean(oof[both].argmax(1) == R['base'][both].argmax(1)):.4f}, "
      f"aka vs v3b {np.mean(oof[both].argmax(1) == R['v3b'][both].argmax(1)):.4f}")
for w in (0.2, 0.35, 0.5):
    la = to_log(oof); la[np.isnan(la)] = 0.0          # aka missing -> base only
    B = norm_probs(to_log(R["base"]) + w * la).astype(np.float32)
    od = os.path.join(HERE, f"bl_{tag}_w{w}"); os.makedirs(od, exist_ok=True); np.save(os.path.join(od, "oof.npy"), B)
    eval_single(B, name=f"base + {w}*{tag}")
eval_single(R["base"], name="base (e7 OOF)")
# test agreement with the akhyar public submission
tp = os.path.join(HERE, f"{tag}_full", "test.npy")
if os.path.exists(tp):
    T = np.load(tp); sub = pd.read_csv(r"E:\Claude code\wear\public_subs\akhyar2612__0-670\submission.csv")
    assert np.all(sub.id.to_numpy() == np.arange(len(T)))
    a = sub.target_feature.to_numpy(); p = T.argmax(1)
    print(f"TEST argmax agreement with akhyar2612/0-670 submission: {np.mean(p == a):.4f} (n={len(a)})")
    tm = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
    for s in sorted(tm.sbj_id.unique()):
        v = tm.sbj_id.to_numpy() == s; print(f"  sbj {s}: {np.mean(p[v] == a[v]):.4f}")
    for loc in sorted(tm.sensor_location.unique()):
        v = tm.sensor_location.to_numpy() == loc; print(f"  {loc}: {np.mean(p[v] == a[v]):.4f}")
    print("pred null frac", np.mean(p == 0).round(4), "akhyar null frac", np.mean(a == 0).round(4))
