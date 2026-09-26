"""sklearn-free version of refine.py's mrf4 path (Windows application control now blocks sklearn.neighbors /
linear_model). Identical math: mrf4 decode -> within-subject kNN (cosine, k=5) label spreading of decoded labels over
build_X(F,'v768',128,32) -> L += w*log Q -> mrf4.
python refine_ns.py --probs <probs.npy> --out <csv> [--w 6] [--variant mrf4]"""
import os, sys, argparse, pickle, time
import numpy as np, pandas as pd, scipy.sparse as sp
TD = r"E:\Claude code\wear\exp\transductive"; DEC = r"E:\Claude code\wear\exp\decoder"
sys.path.insert(0, TD); sys.path.insert(0, DEC)
from tlib import window_features, build_X, NC, DATA, PREP, WORK
from decoder import decode_subject, VARIANTS
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]

def knn_graph_np(X, k):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); n = len(Xn)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    sim = np.clip(np.take_along_axis(S, idx, 1), 0, None)
    W = sp.csr_matrix((sim.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T); dg = np.asarray(W.sum(1)).ravel() + 1e-9
    Dm = sp.diags(1 / np.sqrt(dg)); return Dm @ W @ Dm

def knn_label_Q(X, lab, k=5, alpha=0.9, eps=0.1, iters=30):
    n = len(lab); S = knn_graph_np(X, k); Y0 = np.zeros((n, NC)); Y0[np.arange(n), lab] = 1; F = Y0.copy()
    for _ in range(iters): F = alpha * (S @ F) + (1 - alpha) * Y0
    Q = F / np.maximum(F.sum(1, keepdims=True), 1e-12); return (1 - eps) * Q + eps / NC

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--probs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=6.0); ap.add_argument("--variant", default="mrf4")
    ap.add_argument("--struct", default=os.path.join(WORK, "test_structure.pkl"))
    a = ap.parse_args(); cfg = VARIANTS[a.variant]
    P_all = np.load(a.probs).astype(np.float64); P_all /= P_all.sum(1, keepdims=True)
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)
    vp = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r"); v768 = np.load(os.path.join(PREP, "test_vid_mean768.npy"), mmap_mode="r")
    limb = np.array([LIMBS.index(l) for l in tm.sensor_location]); struct = pickle.load(open(a.struct, "rb"))
    lab = P_all.argmax(1).copy()
    for s, st in struct.items():
        t0 = time.time(); idx = st["idx"]; P = P_all[idx]
        lab0 = decode_subject(P, st, cfg, None)
        F = window_features(vp[idx], v768[idx], xi[idx], limb[idx]); X = build_X(F, "v768", 128, 32)
        L = np.log(np.clip(P, 1e-6, 1)) + a.w * np.log(knn_label_Q(X, lab0))
        Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        l = decode_subject(Pn, st, cfg, None); lab[idx] = l
        print(f"sbj {s}: changed vs mrf4 {np.mean(l != lab0):.3f} null {np.mean(l==0):.3f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame({"id": tm.id, "target_feature": lab}).to_csv(a.out, index=False); print("wrote", a.out, "null", round(float((lab == 0).mean()), 3))

if __name__ == "__main__":
    main()
