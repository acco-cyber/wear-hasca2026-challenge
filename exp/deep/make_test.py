"""Assemble the deep window model: log-blend of video transformer, IMU net, pose probe with fold-0 weights.
Writes exp/deep/deep_full/{test.npy, oof_f0.npy, experts_test.npz, info.json}; compares test argmax to our best sub.
python make_test.py --vid dirA dirB --imu dirC dirD --probe fileE --probe_f0 fileF --w 0.1 0.4 0.5"""
import os, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
ap = argparse.ArgumentParser()
ap.add_argument("--vid", nargs="+", required=True); ap.add_argument("--imu", nargs="+", required=True)
ap.add_argument("--probe", required=True); ap.add_argument("--probe_f0", required=True)
ap.add_argument("--vid_f0", required=True); ap.add_argument("--imu_f0", required=True)
ap.add_argument("--w", nargs=3, type=float, required=True); a = ap.parse_args()
EXPD = r"E:\Claude code\wear\exp\deep"; OD = os.path.join(EXPD, "deep_full"); os.makedirs(OD, exist_ok=True); NC = 19
def lg(P): return np.log(np.clip(P, 1e-6, 1))
def norm(L):
    L = L - np.nanmax(L, -1, keepdims=True); P = np.exp(L); return (P / np.nansum(P, -1, keepdims=True)).astype(np.float32)
def avg(dirs):  # "dir" or "dir:weight"
    ws = [float(d.split(":")[1]) if ":" in d else 1.0 for d in dirs]
    return sum(w * np.load(os.path.join(EXPD, d.split(":")[0], "test.npy")) for d, w in zip(dirs, ws)) / sum(ws)
V, I, Pp = avg(a.vid), avg(a.imu), np.load(a.probe)
T = norm(a.w[0] * lg(V) + a.w[1] * lg(I) + a.w[2] * lg(Pp))
assert T.shape == (12234, NC) and np.isfinite(T).all()
np.save(os.path.join(OD, "test.npy"), T); np.savez_compressed(os.path.join(OD, "experts_test.npz"), vid=V, imu=I, probe=Pp)
# fold-0 holdout combined (N,4,19), NaN outside fold 0 / invalid limbs
O = [np.load(p).astype(np.float32) for p in (a.vid_f0, a.imu_f0, a.probe_f0)]
F0 = norm(a.w[0] * lg(O[0]) + a.w[1] * lg(O[1]) + a.w[2] * lg(O[2]))
F0[np.isnan(O[1][..., 0])] = np.nan
np.save(os.path.join(OD, "oof_f0.npy"), F0)
# compare with our best submission
best = pd.read_csv(r"E:\Claude code\wear\subs\sub_transductive_mrf4_e19_aka045_abh.csv").sort_values("id")
tm = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
yb = best.target_feature.to_numpy(); yd = T.argmax(1); info = {"weights": a.w, "vid": a.vid, "imu": a.imu, "probe": a.probe}
def cmp(tag, yp):
    r = {"agree": float((yp == yb).mean()), "macroF1_vs_best": float(f1_score(yb, yp, average="macro")),
         "per_sbj_agree": {int(s): round(float((yp == yb)[tm.sbj_id == s].mean()), 4) for s in np.unique(tm.sbj_id)}}
    print(tag, json.dumps(r)); return r
info["deep"] = cmp("deep blend", yd)
for n, X in (("vid", V), ("imu", I), ("probe", Pp)): info[n] = cmp(n, X.argmax(1))
for p, n in ((r"E:\Claude code\wear\work\lgbm_v1\test.npy", "lgbm_v1"), (r"E:\Claude code\wear\exp\base\v3b\test.npy", "v3b")):
    if os.path.exists(p): info[n] = cmp(n, np.load(p).argmax(1))
print("deep test pred dist", np.bincount(yd, minlength=NC).tolist())
print("best sub dist      ", np.bincount(yb, minlength=NC).tolist())
json.dump(info, open(os.path.join(OD, "info.json"), "w"), indent=1)
