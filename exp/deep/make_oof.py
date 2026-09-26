"""Assemble 5-fold OOF (N,4,19) of the deep blend from per-fold expert holdouts; report overall + per-fold scores
and the blend with our base OOF. Writes exp/deep/deep_full/oof.npy (+ experts_oof.npz)."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
EXPD = r"E:\Claude code\wear\exp\deep"; PREP = r"E:\Claude code\wear\data\prep"; WORK = r"E:\Claude code\wear\work"; NC = 19
W = [0.2, 0.4, 0.4]
m = pd.read_csv(os.path.join(PREP, "train_meta.csv")); N = len(m); y = m.y.to_numpy(); pur = m.pur.to_numpy(); sbj = m.sbj.to_numpy()
perm = np.random.RandomState(0).permutation(np.unique(sbj)); fo = {s: i % 5 for i, s in enumerate(perm)}; fold = np.array([fo[s] for s in sbj])
ref = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy"), mmap_mode="r"); refv = ~np.isnan(np.asarray(ref[:, :, 0])); rng5 = np.random.RandomState(5)
RL = np.array([rng5.choice(np.where(v)[0]) if v.any() else -1 for v in refv]); del ref
ok = (RL >= 0) & (pur >= 0.8)
def lg(P): return np.log(np.clip(P, 1e-6, 1))
def gather(fn):
    O = np.full((N, 4, NC), np.nan, np.float32); have = []
    for k in range(5):
        p = fn(k)
        if os.path.exists(p):
            o = np.load(p); O[fold == k] = o[fold == k]; have.append(k)
    return O, have
V, hv = gather(lambda k: os.path.join(EXPD, f"vid_f{k}_s0_c1e4", f"oof_f{k}.npy"))
I, hi = gather(lambda k: os.path.join(EXPD, f"imu_f{k}_s0", f"oof_f{k}.npy"))
P = np.load(os.path.join(EXPD, "probe_cv5_c1", "oof_C0.001.npy"))
print("folds with vid", hv, "imu", hi)
L = W[0] * lg(V) + W[1] * lg(I) + W[2] * lg(P); L = L - L.max(-1, keepdims=True); D = np.exp(L); D /= D.sum(-1, keepdims=True)
D = D.astype(np.float32); D[np.isnan(I[..., 0]) | np.isnan(V[..., 0])] = np.nan
def sc(X, mask):
    idx = np.where(mask)[0]; Q = X[idx, RL[idx]]; Q = np.where(np.isnan(Q[:, :1]), 1.0 / NC, Q)
    return f1_score(y[idx], Q.argmax(1), average="macro")
done = [k for k in range(5) if k in hv and k in hi]
dm = ok & np.isin(fold, done)
for n, X in (("vid", V), ("imu", I), ("probe", P), ("deep", D)):
    print(f"{n:6s} OOF single-limb F1 over folds {done}: {sc(X, dm):.4f} | per fold", [round(sc(X, ok & (fold == k)), 4) for k in done])
B = np.load(os.path.join(r"E:\Claude code\wear\exp\base", "bl_v3b_v1_f", "oof.npy"))
print("bl_v3b_v1_f", round(sc(B, dm), 4), "| per fold", [round(sc(B, ok & (fold == k)), 4) for k in done])
for wd in (0.3, 0.5, 0.6, 0.7, 0.8):
    Lb = (1 - wd) * lg(B) + wd * lg(np.where(np.isnan(D), 1.0 / NC, D)); Lb = Lb - Lb.max(-1, keepdims=True); Eb = np.exp(Lb); Eb /= Eb.sum(-1, keepdims=True)
    Eb[np.isnan(B[..., 0])] = np.nan
    print(f"  base + deep w={wd}: {sc(Eb, dm):.4f} | per fold", [round(sc(Eb, ok & (fold == k)), 4) for k in done])
if len(done) == 5:
    os.makedirs(os.path.join(EXPD, "deep_full"), exist_ok=True)
    np.save(os.path.join(EXPD, "deep_full", "oof.npy"), D)
    np.savez_compressed(os.path.join(EXPD, "deep_full", "experts_oof.npz"), vid=V, imu=I, probe=P)
    print("wrote deep_full/oof.npy")
