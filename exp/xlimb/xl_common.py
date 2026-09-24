"""Cross-limb link evidence: per-window IMU descriptors, cross-limb ridge continuity models (fit on train seconds where
all four limbs are recorded simultaneously), and augmented pair features for the successor scorer."""
import os, sys, pickle
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import pair_features, assignment

ROOT = r"E:\Claude code\wear"; DATA = os.path.join(ROOT, "data"); PREP = os.path.join(DATA, "prep"); WORK = os.path.join(ROOT, "work")
TD = os.path.join(ROOT, "exp", "transductive"); XD = os.path.join(ROOT, "exp", "xlimb")
FIT = ["sbj_1", "sbj_3", "sbj_7", "sbj_12", "sbj_16", "sbj_19"]
SESS = {"eval": ["sbj_0", "sbj_5", "sbj_10", "sbj_14_2", "sbj_20", "sbj_21"],
        "extra": ["sbj_2", "sbj_4", "sbj_8", "sbj_9", "sbj_13", "sbj_17"],
        "extra2": ["sbj_0_2", "sbj_6", "sbj_11", "sbj_14", "sbj_15", "sbj_18"]}
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
FS = 50.0

def load_prep():
    meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    imu = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r")
    vid = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
    sl = {s: (int(g.index.min()), int(g.index.max()) + 1) for s, g in meta.groupby("session", sort=False)}
    return meta, imu, vid, sl

# ------------------------------------------------------------------ per-window descriptor
DESC_NAMES = (["gx", "gy", "gz", "ox", "oy", "oz", "sdx", "sdy", "sdz", "lmag_mu", "lmag_sd", "hx", "hy", "hz", "tx", "ty", "tz",
               "hvx", "hvy", "hvz", "tvx", "tvy", "tvz", "fdom", "lb0", "lb1", "lb2", "lb3"])

def window_desc(W):
    """W (n,50,3) -> (n,28) float32; NaN rows stay NaN."""
    W = np.asarray(W, np.float32); bad = np.isnan(W).any(axis=(1, 2)); Wc = np.where(bad[:, None, None], 0.0, W)
    g = Wc.mean(1); o = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6); sd = Wc.std(1)
    mag = np.linalg.norm(Wc, axis=2); lmu = np.log(mag.mean(1) + 1e-3); lsd = np.log(mag.std(1) + 1e-3)
    h = Wc[:, :5].mean(1); t = Wc[:, -5:].mean(1); hv = Wc[:, 5:10].mean(1) - h; tv = t - Wc[:, -10:-5].mean(1)
    d = mag - mag.mean(1, keepdims=True); P = np.abs(np.fft.rfft(d, axis=1)) ** 2; f = np.fft.rfftfreq(W.shape[1], 1 / FS)
    P = P[:, 1:]; f = f[1:]; dom = f[P.argmax(1)]
    bands = [np.log1p(P[:, m].sum(1)) for m in (f < 2, (f >= 2) & (f < 4), (f >= 4) & (f < 8), f >= 8)]
    D = np.concatenate([g, o, sd, lmu[:, None], lsd[:, None], h, t, hv, tv, dom[:, None], np.stack(bands, 1)], 1).astype(np.float32)
    D[bad] = np.nan
    return D

def raw_tail(W, k=15):
    W = np.asarray(W, np.float32); return W[:, -k:].reshape(len(W), -1)

def raw_head(W, k=5):
    W = np.asarray(W, np.float32); return W[:, :k].reshape(len(W), -1)

# ------------------------------------------------------------------ cross-limb continuity models
class XLModels:
    """For each ordered limb pair (a,b): ridge predicting [desc_b(t+1), head_b(t+1)] from [desc_a(t), tail_a(t)].
    Residuals standardised per output dim (on the fit data). Also a same-second model desc_b(t) from desc_a(t)."""
    def __init__(self, lam=10.0): self.lam = lam; self.m = {}

    @staticmethod
    def _in(D, T): return np.concatenate([D, T], 1)
    @staticmethod
    def _out(D, H): return np.concatenate([D, H], 1)

    def fit(self, imu, sl, sessions):
        Xs = {l: [] for l in range(4)}; Ys = {l: [] for l in range(4)}; Zs = {l: [] for l in range(4)}
        for s in sessions:
            a, b = sl[s]; W = np.asarray(imu[a:b], np.float32)
            for l in range(4):
                D = window_desc(W[:, l]); Xs[l].append(self._in(D, raw_tail(W[:, l]))[:-1]); Ys[l].append(self._out(D, raw_head(W[:, l]))[1:])
                Zs[l].append(D)
        X = {l: np.concatenate(Xs[l]) for l in range(4)}; Y = {l: np.concatenate(Ys[l]) for l in range(4)}; Z = {l: np.concatenate(Zs[l]) for l in range(4)}
        self.mu_in = {l: np.nanmean(X[l], 0) for l in range(4)}; self.sd_in = {l: np.nanstd(X[l], 0) + 1e-3 for l in range(4)}
        self.mu_d = {l: np.nanmean(Z[l], 0) for l in range(4)}; self.sd_d = {l: np.nanstd(Z[l], 0) + 1e-3 for l in range(4)}
        for la in range(4):
            for lb in range(4):
                for kind, (Xa, Yb) in (("next", (X[la], Y[lb])), ("same", (Z[la], Z[lb]))):
                    ok = ~(np.isnan(Xa).any(1) | np.isnan(Yb).any(1))
                    xin = (Xa[ok] - Xa[ok].mean(0)) / (Xa[ok].std(0) + 1e-3); mx, sx = Xa[ok].mean(0), Xa[ok].std(0) + 1e-3
                    my = Yb[ok].mean(0); yc = Yb[ok] - my
                    A = xin.T @ xin + self.lam * np.eye(xin.shape[1]); Wt = np.linalg.solve(A, xin.T @ yc)
                    res = yc - xin @ Wt; rs = res.std(0) + 1e-4
                    self.m[(kind, la, lb)] = (mx, sx, my, Wt.astype(np.float32), rs, np.float32(yc.std(0).mean()))
        return self

    def predict(self, kind, la, lb, Xa):
        mx, sx, my, Wt, rs, _ = self.m[(kind, la, lb)]
        return ((Xa - mx) / sx) @ Wt + my, rs

# ------------------------------------------------------------------ pair features
NEW_NAMES = ["la", "lb", "same_side", "same_type", "xl_next", "xl_next_d", "xl_next_h", "xl_next_g", "xl_next_e", "xl_same", "xl_same_g",
             "xl_same_e", "base_next", "xl_gain"] + [f"i_{n}" for n in ["ox", "oy", "oz", "lmag_sd", "fdom", "lb1", "lb2"]] + \
            [f"j_{n}" for n in ["ox", "oy", "oz", "lmag_sd", "fdom", "lb1", "lb2"]]
KEEP = [DESC_NAMES.index(n) for n in ["ox", "oy", "oz", "lmag_sd", "fdom", "lb1", "lb2"]]
ND = len(DESC_NAMES)

def new_pair_features(W, limb, cand, xl):
    """W (n,50,3) window of its own limb; limb (n,); cand (n,M). Returns (n,M,len(NEW_NAMES)) float32 (NaN where cand<0)."""
    n, M = cand.shape; limb = np.asarray(limb); valid = cand >= 0; cc = np.where(valid, cand, 0)
    D = window_desc(W); Xin = np.concatenate([D, raw_tail(W)], 1); Yout = np.concatenate([D, raw_head(W)], 1)
    la = np.repeat(limb[:, None], M, 1); lb = limb[cc]
    out = np.full((n, M, len(NEW_NAMES)), np.nan, np.float32)
    out[..., 0] = la; out[..., 1] = lb
    out[..., 2] = ((la % 2) == (lb % 2)) * 0 + ((la < 2) == (lb < 2))          # same side: left(0,1) vs right(2,3)
    out[..., 3] = ((la % 2) == (lb % 2))                                        # same type: arm(0,2) vs leg(1,3)
    rows = np.arange(n)
    gi = [DESC_NAMES.index(x) for x in ("ox", "oy", "oz")]; ei = [DESC_NAMES.index(x) for x in ("lmag_mu", "lmag_sd", "lb0", "lb1", "lb2", "lb3")]
    for a in range(4):
        ia = rows[limb == a]
        if len(ia) == 0: continue
        for b in range(4):
            m = (limb[cc[ia]] == b) & valid[ia]
            if not m.any(): continue
            ii, kk = np.nonzero(m); src = ia[ii]; dst = cc[src, kk]
            pred, rs = xl.predict("next", a, b, Xin[src]); r = (Yout[dst] - pred) / rs
            r2 = r ** 2
            out[src, kk, 4] = np.log1p(np.nanmean(r2, 1)); out[src, kk, 5] = np.log1p(np.nanmean(r2[:, :ND], 1))
            out[src, kk, 6] = np.log1p(np.nanmean(r2[:, ND:], 1)); out[src, kk, 7] = np.log1p(np.nanmean(r2[:, gi], 1))
            out[src, kk, 8] = np.log1p(np.nanmean(r2[:, ei], 1))
            ps, rs2 = xl.predict("same", a, b, D[src]); r = ((D[dst] - ps) / rs2) ** 2
            out[src, kk, 9] = np.log1p(np.nanmean(r, 1)); out[src, kk, 10] = np.log1p(np.nanmean(r[:, gi], 1)); out[src, kk, 11] = np.log1p(np.nanmean(r[:, ei], 1))
            # baseline: residual of the unconditional (mean) prediction -> how much a's window explains
            my = xl.m[("next", a, b)][2]; rsn = xl.m[("next", a, b)][4]
            rb = ((Yout[dst] - my) / rsn) ** 2; out[src, kk, 12] = np.log1p(np.nanmean(rb, 1))
    out[..., 13] = out[..., 12] - out[..., 4]
    Di = D[:, KEEP]; out[..., 14:14 + len(KEEP)] = Di[:, None, :]; out[..., 14 + len(KEEP):] = Di[cc]
    out[~valid] = np.nan
    return out

def session_windows(imu, a, b, limb):
    n = b - a; W4 = np.asarray(imu[a:b], np.float32); return W4[np.arange(n), limb]

def all_pair_features(V, W, limb, xl):
    cand, F0 = pair_features(V, W, limb)
    F1 = new_pair_features(W, limb, cand, xl)
    return cand, np.concatenate([F0, F1], 2)
