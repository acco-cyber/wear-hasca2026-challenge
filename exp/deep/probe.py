"""Pose probe: multinomial logistic regression on standardized [mean, std, mean |frame diff|] of the PCA-160 clip.
python probe.py <f0|full> [--C 0.05]"""
import os, sys, time, argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.05, 0.2])
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--center", type=int, default=0); a = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(a.threads); os.environ["OPENBLAS_NUM_THREADS"] = str(a.threads); os.environ["MKL_NUM_THREADS"] = str(a.threads)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
PREP = r"E:\Claude code\wear\data\prep"; WORK = r"E:\Claude code\wear\work"; EXPD = r"E:\Claude code\wear\exp\deep"; NC = 19
t0 = time.time()
m = pd.read_csv(os.path.join(PREP, "train_meta.csv")); N = len(m); y = m.y.to_numpy(); pur = m.pur.to_numpy(); sbj = m.sbj.to_numpy()
perm = np.random.RandomState(0).permutation(np.unique(sbj)); fo = {s: i % 5 for i, s in enumerate(perm)}; fold = np.array([fo[s] for s in sbj])
ref = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy"), mmap_mode="r"); refv = ~np.isnan(np.asarray(ref[:, :, 0])); rng5 = np.random.RandomState(5)
RL = np.array([rng5.choice(np.where(v)[0]) if v.any() else -1 for v in refv]); del ref
EVAL = np.where((RL >= 0) & (pur >= 0.8) & (fold == 0))[0]

def feats(V):
    V = V.astype(np.float32)
    return np.concatenate([V.mean(1), V.std(1), np.abs(np.diff(V, axis=1)).mean(1)], 1)
F = feats(np.load(os.path.join(PREP, "train_vid_pca.npy"))); Ft = feats(np.load(os.path.join(PREP, "test_vid_pca.npy")))
if a.center:  # per-subject standardization (train subjects and each test subject separately)
    tsb = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv").sbj_id.to_numpy()
    for FF, SS in ((F, sbj), (Ft, tsb)):
        for s in np.unique(SS):
            k = SS == s; FF[k] -= np.median(FF[k], 0)
            if a.center == 2: FF[k] /= (np.percentile(FF[k], 75, 0) - np.percentile(FF[k], 25, 0) + 1e-3)
def prep(trmask):
    tr = np.where(trmask & (pur >= 0.8))[0]
    rs = np.random.RandomState(0)
    # null cap per subject (2.5x largest activity class), sqrt-balanced weights
    keep = []
    for s in np.unique(sbj[tr]):
        g = tr[sbj[tr] == s]; nul = g[y[g] == 0]; act = g[y[g] != 0]; cap = int(round(2.5 * np.bincount(y[act], minlength=NC)[1:].max()))
        keep += [act, rs.choice(nul, cap, replace=False) if len(nul) > cap else nul]
    tr = np.sort(np.concatenate(keep)); cnt = np.bincount(y[tr], minlength=NC).astype(float); cw = cnt ** -0.5; cw = cw / (cw * cnt).sum() * cnt.sum()
    mu = F[tr].mean(0); sd = F[tr].std(0) + 1e-6
    return tr, cw, np.clip((F - mu) / sd, -8, 8), np.clip((Ft - mu) / sd, -8, 8)
OD = os.path.join(EXPD, f"probe_{a.mode}" + (f"_c{a.center}" if a.center else "")); os.makedirs(OD, exist_ok=True); res = {}
if a.mode == "cv5":  # full 5-fold subject OOF (N,4,19) + fold-averaged test
    for C in a.C:
        full = np.full((N, 4, NC), np.nan, np.float32); T = 0
        for k in range(5):
            tr, cw, Z, Zt = prep(fold != k); lr = LogisticRegression(C=C, max_iter=300, tol=1e-3).fit(Z[tr], y[tr], sample_weight=cw[y[tr]])
            va = np.where(fold == k)[0]; full[va] = lr.predict_proba(Z[va]).astype(np.float32)[:, None, :]; T = T + lr.predict_proba(Zt) / 5
        ok = np.where((RL >= 0) & (pur >= 0.8))[0]; f = f1_score(y[ok], full[ok, RL[ok]].argmax(1), average="macro")
        print(f"[{time.time()-t0:.0f}s] probe C={C} 5-fold OOF single-limb F1 {f:.4f}", flush=True); res[f"C{C}"] = f
        np.save(os.path.join(OD, f"oof_C{C}.npy"), full); np.save(os.path.join(OD, f"test_C{C}.npy"), T.astype(np.float32))
    json.dump(res, open(os.path.join(OD, "res.json"), "w"), indent=1); sys.exit(0)
tr, cw, Z, Zt = prep((fold != 0) if a.mode == "f0" else np.ones(N, bool))
for C in a.C:
    lr = LogisticRegression(C=C, max_iter=300, tol=1e-3)
    lr.fit(Z[tr], y[tr], sample_weight=cw[y[tr]])
    tag = f"C{C}"
    if a.mode == "f0":
        va = np.where(fold == 0)[0]; P = lr.predict_proba(Z[va]).astype(np.float32)
        full = np.full((N, 4, NC), np.nan, np.float32); full[va] = P[:, None, :]
        f = f1_score(y[EVAL], full[EVAL, RL[EVAL]].argmax(1), average="macro"); res[tag] = f
        np.save(os.path.join(OD, f"oof_f0_{tag}.npy"), full)
        print(f"[{time.time()-t0:.0f}s] probe C={C} fold0 single-limb F1 {f:.4f}", flush=True)
    np.save(os.path.join(OD, f"test_{tag}.npy"), lr.predict_proba(Zt).astype(np.float32))
json.dump(res, open(os.path.join(OD, "res.json"), "w"), indent=1); print("done", flush=True)
