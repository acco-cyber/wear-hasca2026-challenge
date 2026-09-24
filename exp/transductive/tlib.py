"""Shared helpers for per-subject transductive refinement (sim + test)."""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import cut, chains_from_succ
from decode import calibrate_counts, build_graph, graph_smooth, segments
from imu_feats import limb_features

ROOT = r"E:\Claude code\wear"; DATA = os.path.join(ROOT, "data"); PREP = os.path.join(DATA, "prep"); WORK = os.path.join(ROOT, "work")
TD = os.path.join(ROOT, "exp", "transductive")
NC = 19
EVAL = ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"]
EXTRA = ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"]
EXTRA2 = ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]

def blend(P1, P2, w2=0.2):
    L = (1 - w2) * np.log(np.clip(P1, 1e-6, 1)) + w2 * np.log(np.clip(P2, 1e-6, 1))
    P = np.exp(L - np.nanmax(L, -1, keepdims=True)); return P / np.nansum(P, -1, keepdims=True)

def load_oof_blend():
    o1 = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy")); o2 = np.load(os.path.join(WORK, "fusion_v1", "oof.npy"))
    L = 0.8 * np.log(np.clip(o1, 1e-6, 1)) + 0.2 * np.log(np.clip(o2, 1e-6, 1))
    P = np.exp(L); P /= np.nansum(P, 2, keepdims=True); return P      # identical to sim_decode.py blend

def load_structs(which="eval"):
    if which == "eval": S = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb"))
    else: S = pickle.load(open(os.path.join(TD, f"{which}_struct.pkl"), "rb"))
    for s in S: S[s]["Lm"] = np.asarray(S[s]["Lm"], np.float32)
    return S

def session_P(oof, st):
    n = st["n"]; P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]]; ok = ~np.isnan(P[:, 0])
    return np.where(ok[:, None], P, 1.0 / NC)

# ---------------------------------------------------------------- features
def window_features(vpca, v768, imu_w, limb):
    """vpca (n,15,160), v768 (n,768), imu_w (n,50,3) of the window's own limb, limb (n,) -> dict of blocks."""
    V = np.asarray(vpca, np.float32); v768 = np.asarray(v768, np.float32)
    vm = V.mean(1); vs = V.std(1); vd = V[:, -3:].mean(1) - V[:, :3].mean(1)
    fi = limb_features(np.asarray(imu_w, np.float32))
    fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(vm=vm, vs=vs, vd=vd, v768=v768, imu=fi, limb=np.asarray(limb))

def std_within(X, groups=None):
    X = np.asarray(X, np.float64); out = np.empty_like(X)
    if groups is None: groups = np.zeros(len(X), int)
    for g in np.unique(groups):
        m = groups == g; mu = X[m].mean(0); sd = X[m].std(0) + 1e-6; out[m] = (X[m] - mu) / sd
    return out

def pca_reduce(X, d):
    X = X - X.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:d].T; return Z / (S[:d] / np.sqrt(len(X)) + 1e-9)          # whitened

def build_X(F, spec="vm+vs+vd+imu", d_vid=64, d_imu=32):
    """Per-subject standardised + PCA-whitened design matrix. IMU standardised within (subject, limb)."""
    parts = []
    vb = [k for k in ("vm", "vs", "vd", "v768") if k in spec.split("+")]
    if vb:
        Xv = std_within(np.concatenate([F[k] for k in vb], 1)); parts.append(pca_reduce(Xv, d_vid))
    if "imu" in spec.split("+"):
        Xi = std_within(F["imu"], F["limb"]); Xi = np.clip(Xi, -6, 6); parts.append(pca_reduce(Xi, d_imu))
    if "lp" in spec.split("+"):          # base-model log-probabilities (subject-standardised): per-subject recalibration
        parts.append(np.clip(std_within(F["lp"]), -6, 6))
    return np.concatenate(parts, 1)

# ---------------------------------------------------------------- baseline decoder
def prep_decoder(st, n):
    g = build_graph(st["cand"], st["lo"], n, k=10)
    chains = chains_from_succ(cut(st["succ0"], st["sc"], np.asarray(st["Lm"], np.float32), -6.0))
    return g, chains

DEC = dict(null_scale=0.5, lo=80, hi=250, p_stay=0.8)     # lead's updated baseline (sim 0.7543, LB 0.817); old: 1.0/60/160

def decode(P, g, chains, dec=None):
    d = {**DEC, **(dec or {})}
    P = P.copy(); P[:, 0] *= d["null_scale"]; P /= P.sum(1, keepdims=True)
    Pg = graph_smooth(P, g, alpha=0.5, iters=5)
    lab, b = calibrate_counts(Pg, chains, lo=d["lo"], hi=d["hi"], p_stay=d["p_stay"])
    return lab, Pg, b
