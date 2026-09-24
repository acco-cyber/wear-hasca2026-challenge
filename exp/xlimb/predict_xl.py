"""e7 recipe on TEST with an alternative link structure: per subject mrf4 -> within-subject video-kNN label spreading
(v768 d128, k5, alpha .9, eps .1, L += 6 log Q) -> mrf4.   python predict_xl.py <probs.npy> <struct.pkl> <out.csv>"""
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, pandas as pd
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"
sys.path.insert(0, TD); sys.path.insert(0, DEC)
from tlib import window_features, build_X, NC
from refine_core import knn_label_Q
from decoder import decode_subject, VARIANTS
DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep")
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]

if __name__ == "__main__":
    probs, spath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    P_all = np.load(probs).astype(np.float64); P_all /= P_all.sum(1, keepdims=True); assert P_all.shape == (12234, NC)
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)
    vp = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r"); v768 = np.load(os.path.join(PREP, "test_vid_mean768.npy"), mmap_mode="r")
    limb = np.array([LIMBS.index(l) for l in tm.sensor_location]); struct = pickle.load(open(spath, "rb"))
    cfg = VARIANTS["mrf4"]; lab = P_all.argmax(1).copy()
    for s, st in struct.items():
        t0 = time.time(); st = dict(st); st["Lm"] = np.asarray(st["Lm"], np.float32); idx = st["idx"]; n = len(idx)
        P = P_all[idx]; L = np.log(np.clip(P, 1e-6, 1)); lab0 = decode_subject(P, st, cfg, None)
        F = window_features(vp[idx], v768[idx], xi[idx], limb[idx]); X = build_X(F, "v768", 128, 32)
        L = L + 6.0 * np.log(knn_label_Q(X, lab0, 5, 0.9, 0.1)); Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        l = decode_subject(Pn, st, cfg, None); lab[idx] = l
        print(f"sbj {s}: n={n} null {np.mean(l == 0):.3f} changed-by-refine {np.mean(l != lab0):.3f} classes {np.unique(l).size} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame({"id": tm.id, "target_feature": lab}).to_csv(out, index=False); print("wrote", out, "null", round(float((lab == 0).mean()), 3))
