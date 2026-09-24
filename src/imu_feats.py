"""Vectorised per-limb IMU features for 1-s windows (50 x 3). Streams: x, y, z, |a|.
Per stream 26 features (13 time + 8 freq + 5 sub-window) + 3 axis correlations + 3 gravity-angle feats = 110.
NaN windows (dropped sensor) -> all-NaN row (LightGBM handles NaN)."""
import numpy as np

FS = 50.0

def _time_feats(S):
    # S: (n,T)
    mu = S.mean(1); sd = S.std(1)
    d = S - mu[:, None]
    m2 = (d ** 2).mean(1) + 1e-12
    skew = (d ** 3).mean(1) / m2 ** 1.5
    kurt = (d ** 4).mean(1) / m2 ** 2 - 3.0
    rms = np.sqrt((S ** 2).mean(1))
    mn = S.min(1); mx = S.max(1); ptp = mx - mn
    med = np.median(S, 1)
    q1, q3 = np.percentile(S, [25, 75], axis=1)
    zc = (np.diff(np.sign(d), axis=1) != 0).mean(1)
    dx = np.diff(S, axis=1)
    jerk = dx.std(1); jmax = np.abs(dx).max(1)
    ac1 = (d[:, 1:] * d[:, :-1]).mean(1) / m2
    return [mu, sd, rms, mn, mx, ptp, med, q3 - q1, skew, kurt, zc, jerk, jmax, ac1]

def _freq_feats(S):
    n, T = S.shape
    d = S - S.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(d, axis=1)) ** 2
    f = np.fft.rfftfreq(T, 1 / FS)
    P = P[:, 1:]; f = f[1:]                       # drop DC
    tot = P.sum(1) + 1e-12
    Pn = P / tot[:, None]
    dom = f[P.argmax(1)]
    cent = (Pn * f).sum(1)
    ent = -(Pn * np.log(Pn + 1e-12)).sum(1)
    b1 = P[:, f < 2].sum(1); b2 = P[:, (f >= 2) & (f < 5)].sum(1); b3 = P[:, (f >= 5) & (f < 10)].sum(1); b4 = P[:, f >= 10].sum(1)
    cum = np.cumsum(Pn, 1); roll = f[np.minimum((cum < 0.95).sum(1), len(f) - 1)]
    return [np.log1p(tot), dom, cent, ent, np.log1p(b1), np.log1p(b2), np.log1p(b3), np.log1p(b4), roll]

def _sub_feats(S):
    h = S.shape[1] // 2
    a, b = S[:, :h], S[:, h:]
    t = np.arange(S.shape[1]); tc = t - t.mean()
    slope = ((S - S.mean(1, keepdims=True)) * tc).sum(1) / (tc ** 2).sum()
    return [b.mean(1) - a.mean(1), b.std(1) - a.std(1), slope]

def limb_features(W):
    """W: (n,50,3) float -> (n,F) float32. Rows with any NaN -> NaN."""
    W = np.asarray(W, np.float32)
    bad = np.isnan(W).any(axis=(1, 2))
    Wc = np.where(bad[:, None, None], 0.0, W)
    mag = np.linalg.norm(Wc, axis=2)
    outs = []
    for S in (Wc[:, :, 0], Wc[:, :, 1], Wc[:, :, 2], mag):
        outs += _time_feats(S) + _freq_feats(S) + _sub_feats(S)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        xa = Wc[:, :, a] - Wc[:, :, a].mean(1, keepdims=True)
        xb = Wc[:, :, b] - Wc[:, :, b].mean(1, keepdims=True)
        outs.append((xa * xb).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1) + 1e-9))
    g = Wc.mean(1); gn = np.linalg.norm(g, axis=1) + 1e-9
    outs += [g[:, 0] / gn, g[:, 1] / gn, g[:, 2] / gn]          # gravity direction (limb orientation)
    F = np.stack(outs, 1).astype(np.float32)
    F[bad] = np.nan
    return F

def feature_names():
    names = []
    for s in ("x", "y", "z", "m"):
        names += [f"{s}_{k}" for k in ["mu", "sd", "rms", "min", "max", "ptp", "med", "iqr", "skew", "kurt", "zc", "jerk", "jmax", "ac1",
                                         "lpow", "fdom", "fcent", "fent", "b0_2", "b2_5", "b5_10", "b10", "roll95", "dmu", "dsd", "slope"]]
    names += ["cxy", "cxz", "cyz", "gx", "gy", "gz"]
    return names

if __name__ == "__main__":
    import time
    x = np.random.randn(1000, 50, 3).astype(np.float32)
    t = time.time(); F = limb_features(x); print(F.shape, len(feature_names()), f"{time.time()-t:.2f}s")
