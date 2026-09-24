"""Final test decode: probs (12234,19) -> per subject: mrf4 -> [kNN label-spread refinement w] -> [session-block prior beta]
-> mrf4 -> submission.  python predict_final.py <probs.npy> <out.csv> [--w_knn 6] [--beta 4] [--no_refine]"""
import os, sys, argparse, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"; PLD = r"E:\Claude code\wear\exp\pl"
sys.path.insert(0, TD); sys.path.insert(0, DEC); sys.path.insert(0, PLD)
from tlib import window_features, build_X, NC
from refine_core import knn_label_Q
from decoder import decode_subject, VARIANTS
from sim_block import block_prob, apply_block, BLOCK_B
ROOT = r"E:\Claude code\wear"; DATA = os.path.join(ROOT, "data"); PREP = os.path.join(DATA, "prep"); WORK = os.path.join(ROOT, "work")
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]

ap = argparse.ArgumentParser(); ap.add_argument("probs"); ap.add_argument("out")
ap.add_argument("--w_knn", type=float, default=6.0); ap.add_argument("--beta", type=float, default=4.0); ap.add_argument("--no_refine", action="store_true")
a = ap.parse_args()
P_all = np.load(a.probs).astype(np.float64); P_all /= P_all.sum(1, keepdims=True)
tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)
vp = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r"); v768 = np.load(os.path.join(PREP, "test_vid_mean768.npy"), mmap_mode="r")
limb = np.array([LIMBS.index(l) for l in tm.sensor_location]); struct = pickle.load(open(os.path.join(WORK, "test_structure.pkl"), "rb"))
cfg = VARIANTS["mrf4"]; lab = P_all.argmax(1).copy()
def dec(L, st):
    Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True); return decode_subject(Pn, st, cfg, None)
for s, st in struct.items():
    t0 = time.time(); idx = st["idx"]; n = len(idx); P = P_all[idx]; L = np.log(np.clip(P, 1e-6, 1))
    lab0 = decode_subject(P, st, cfg, None); l = lab0
    F = window_features(vp[idx], v768[idx], xi[idx], limb[idx]); X = build_X(F, "v768", 128, 32)
    if not a.no_refine:
        L = L + a.w_knn * np.log(knn_label_Q(X, l, 5, 0.9, 0.1)); l = dec(L, st)
    if a.beta > 0:
        pB = block_prob(X, l, "lr", 0.1); L = apply_block(L, pB, a.beta); l2 = dec(L, st)
        print(f"sbj {s}: block B share among activity windows {np.mean(pB[l>0] > 0.5):.3f}; block-prior changed {np.mean(l2 != l):.3f}", flush=True); l = l2
    lab[idx] = l
    print(f"sbj {s}: n={n} changed vs plain mrf4 {np.mean(l != lab0):.3f} null {np.mean(l==0):.3f} counts {np.bincount(l, minlength=NC).tolist()} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame({"id": tm.id, "target_feature": lab}).to_csv(a.out, index=False); print("wrote", a.out, "null", round(float((lab == 0).mean()), 3))
