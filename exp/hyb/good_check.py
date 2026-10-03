"""Sanity checks on the public 0.927 notebook's final probabilities (oof + test) and first combination CVs:
alignment with our sim_meta rows, calibrated or not, gate with our OOF labels, blend with our recipe's OOF P."""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hanbat_stack import finish, macro_f1, KEEP, HB, TRAIN_SETS, N_CLS, calibrate, CFG, _norm_rows
W = r"E:\Claude code\wear"; G = os.path.join(W, "public_src", "good927", "out")
z = np.load(os.path.join(G, "final_probabilities.npz")); O = z["oof"].astype(np.float64); T = z["test"].astype(np.float64)
sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}; y, sbj, fold = sm["y"], sm["sbj"], sm["fold"]
print("oof row sums", O.sum(1)[:3].round(4), "test row sums", T.sum(1)[:3].round(4))
print(f"OOF argmax F1 {macro_f1(y, O.argmax(1)):.4f} | finish(O) F1 {macro_f1(y, finish(_norm_rows(O), dict(sbj=sbj, sets=TRAIN_SETS))):.4f} | per fold argmax "
      + " ".join(f"{macro_f1(y[fold == k], O[fold == k].argmax(1)):.4f}" for k in range(5)))
sub = pd.read_csv(os.path.join(G, "submission.csv")).sort_values("id").target_feature.to_numpy()
bl = np.load(os.path.join(KEEP, "blend.npz")); tsbj = bl["test_sbj"].astype(np.int64)
print("test argmax vs their submission:", np.mean(T.argmax(1) == sub).round(4), "| finish(T) vs submission:", np.mean(finish(_norm_rows(T), dict(sbj=tsbj, sets={})) == sub).round(4))
print("class counts per test subject (argmax):", {int(s): np.bincount(T[tsbj == s].argmax(1), minlength=19).tolist() for s in np.unique(tsbj)})
conf = O.max(1); print("OOF confidence quantiles", np.quantile(conf, [0.05, 0.1, 0.25, 0.5]).round(3), "| test", np.quantile(T.max(1), [0.05, 0.1, 0.25, 0.5]).round(3))
# gate with our OOF decoder labels (18 sessions)
o2t = np.load(os.path.join(W, "exp", "hyb", "rows.npz"))["ours_to_theirs"]; R = pickle.load(open(os.path.join(W, "exp", "pl", "labels_v3bv1f.pkl"), "rb"))
ours = np.full(len(y), -1, np.int64)
for s, d in R.items():
    ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
S = ours >= 0; base = O.argmax(1); srt = np.argsort(-O, 1)
print(f"\nS rows {S.sum()}: their F1(S) {macro_f1(y[S], base[S]):.4f}; ours {macro_f1(y[S], ours[S]):.4f}")
for tau in (0.4, 0.5, 0.6, 0.7):
    for rule in ("plain", "top2"):
        g = S & (conf < tau)
        if rule == "top2":
            g &= (srt[:, 0] == ours) | (srt[:, 1] == ours)
        lab = base.copy(); lab[g] = ours[g]; print(f"  gate {tau} {rule}: F1(S) {macro_f1(y[S], lab[S]):.4f} gated {g.sum() / S.sum():.3f}")
# blend with our recipe OOF P (L2-faithful): work/hanbat/l2base_P.npy (0.9045) and fused variants
for nm in ("l2base_P.npy",):
    p = os.path.join(HB, nm)
    if os.path.exists(p):
        P = np.load(p).astype(np.float64); P /= P.sum(1, keepdims=True)
        Q = np.clip(P, 1e-12, None) ** 2; Q = calibrate(Q / Q.sum(1, keepdims=True), sbj, TRAIN_SETS, CFG["per_ex"], CFG["null_min"])
        print(f"\nblend with {nm}: ours alone (finish) {macro_f1(y, Q.argmax(1)):.4f}")
        for w in (0.1, 0.2, 0.3, 0.5):
            B = _norm_rows(O ** (1 - w) * Q ** w); print(f"  geometric blend w={w}: F1 {macro_f1(y, B.argmax(1)):.4f}")
