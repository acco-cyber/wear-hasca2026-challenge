"""Fast, vectorised feature engineering for the WEAR challenge.

Based on the prior WEAR winners (FAME: frequency-domain features; 1st winner:
rich features + gradient boosting).  All feature extraction is vectorised over
the batch so it runs in seconds, not hours.
"""
from __future__ import annotations

import numpy as np

N_ACC_FEATS = 19  # per axis/magnitude: 13 time + 6 frequency


def _freq_features_vect(x: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """x: (..., N) zero-mean signals -> (..., 8) freq features. Vectorised."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    mag = np.abs(np.fft.rfft(x, axis=-1))          # (..., n//2+1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    power = mag ** 2
    total = power.sum(axis=-1, keepdims=True)
    total = np.where(total < 1e-12, 1.0, total)
    p = power / total

    dom_freq = freqs[power.argmax(axis=-1)]
    centroid = (freqs[None, :] * power).sum(axis=-1) / total[..., 0]
    entropy = -np.sum(p * np.log(p + 1e-12), axis=-1)

    low = power[..., freqs < 2].sum(axis=-1) / total[..., 0]
    mid = power[..., (freqs >= 2) & (freqs < 8)].sum(axis=-1) / total[..., 0]
    high = power[..., freqs >= 8].sum(axis=-1) / total[..., 0]

    # spectral rolloff (95% energy) and spectral flux
    cum = np.cumsum(p, axis=-1)
    rolloff_idx = np.argmax(cum >= 0.95, axis=-1)
    rolloff = freqs[rolloff_idx]
    # spectral flux (first-order difference of magnitude)
    flux = np.mean(np.abs(np.diff(p, axis=-1)), axis=-1)

    return np.stack([dom_freq, centroid, entropy, low, mid, high,
                     rolloff, flux], axis=-1)


def _time_features_vect(x: np.ndarray) -> np.ndarray:
    """x: (..., N) -> (..., 13) time features. Vectorised, NaN-safe."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    m = x.mean(axis=-1)
    # stable std
    s = np.sqrt(np.mean((x - m[..., None]) ** 2, axis=-1) + 1e-12)
    rms = np.sqrt(np.mean(x ** 2, axis=-1) + 1e-12)
    xmin = x.min(axis=-1)
    xmax = x.max(axis=-1)
    ptp = xmax - xmin
    median = np.median(x, axis=-1)
    # skew / kurtosis (central moments, NaN-safe)
    zm = x - m[..., None]
    z2 = np.mean(zm ** 2, axis=-1) + 1e-12
    skew = np.mean(zm ** 3, axis=-1) / (z2 ** 1.5)
    kurt = np.mean(zm ** 4, axis=-1) / (z2 ** 2) - 3.0
    # zero crossings & mean crossings (per sample)
    zc = (np.sign(zm[..., 1:]) * np.sign(zm[..., :-1]) < 0).sum(axis=-1) / max(1, n - 1)
    mcr = (np.sign(x[..., 1:] - m[..., None]) * np.sign(x[..., :-1] - m[..., None]) < 0).sum(axis=-1) / max(1, n - 1)
    # autocorr lag-1
    xm = x - m[..., None]
    num = np.mean(xm[..., :-1] * xm[..., 1:], axis=-1)
    ar1 = num / (np.mean(xm[..., :-1] ** 2, axis=-1) + 1e-12)
    # jerk proxy (std of first difference)
    jerk = np.std(np.diff(x, axis=-1), axis=-1)
    td = np.stack([m, s, rms, xmin, xmax, ptp, median, skew, kurt,
                   zc, mcr, ar1, jerk], axis=-1)
    return np.nan_to_num(td, nan=0.0, posinf=0.0, neginf=0.0)


def _axis_features_vect(x: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """x: (..., N) -> (..., 21) features. Vectorised (13 time + 8 freq)."""
    td = _time_features_vect(x)
    fr = _freq_features_vect(x, fs)
    return np.concatenate([td, fr], axis=-1)


def _subwindow_features_vect(x: np.ndarray) -> np.ndarray:
    """x: (..., N) -> (..., 4) sub-window trend features.

    Splits the window in half and compares summary stats between the two
    halves to capture temporal structure/drift.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    half = max(1, n // 2)
    x0 = x[..., :half]
    x1 = x[..., half:2 * half]
    m0 = x0.mean(axis=-1)
    m1 = x1.mean(axis=-1)
    s0 = x0.std(axis=-1)
    s1 = x1.std(axis=-1)
    # energy ratio first/second half, and trend of mean/std
    e0 = np.mean(x0 ** 2, axis=-1)
    e1 = np.mean(x1 ** 2, axis=-1)
    trend_mean = m1 - m0
    trend_std = s1 - s0
    energy_ratio = e1 / (e0 + 1e-9)
    # slope of linear fit (normalized)
    t = np.linspace(0, 1, n)
    denom = np.sum((t - t.mean()) ** 2)
    slope = np.sum((x - x.mean(axis=-1, keepdims=True)) * (t - t.mean()), axis=-1) / denom
    return np.stack([trend_mean, trend_std, energy_ratio, slope], axis=-1)


def extract_sensor_correlations(windows: np.ndarray) -> np.ndarray:
    """windows: (N, 50, 4, 3) -> (N, 12) pairwise sensor magnitude correlations.

    For each pair of sensors, correlation of their magnitude time-series.
    """
    w = np.asarray(windows, dtype=np.float64)
    N, T, S, A = w.shape
    mag = np.linalg.norm(w, axis=-1)              # (N,T,S)
    out = np.zeros((N, S * (S - 1) // 2), dtype=np.float64)
    k = 0
    for i in range(S):
        for j in range(i + 1, S):
            a = mag[:, :, i] - mag[:, :, i].mean(axis=-1, keepdims=True)
            b = mag[:, :, j] - mag[:, :, j].mean(axis=-1, keepdims=True)
            denom = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)) + 1e-9
            out[:, k] = (a * b).sum(-1) / denom
            k += 1
    return out


def fit_inertial_pca(F_base: np.ndarray, n_components: int = 32,
                     seed: int = 42):
    """Fit PCA on base inertial feature matrix for feature augmentation."""
    Xc = F_base - F_base.mean(axis=0)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=min(n_components, Xc.shape[1] - 1),
                       random_state=seed)
    svd.fit(Xc)
    return Xc.mean(axis=0), svd.components_.T


def apply_inertial_pca(F_base: np.ndarray, center, components) -> np.ndarray:
    return (F_base - center) @ components


def _lowpass(x, alpha=0.1):
    """Exponential moving average along last axis (approx gravity)."""
    out = np.empty_like(x)
    acc = x[..., 0].copy()
    out[..., 0] = acc
    for t in range(1, x.shape[-1]):
        acc = alpha * x[..., t] + (1 - alpha) * acc
        out[..., t] = acc
    return out


def _gravity_body_decompose(a):
    """a: (..., T, 3) -> a_parallel (...,T,1), a_perp (...,T,1), gravity_norm (...,T,1)."""
    a = np.asarray(a, dtype=np.float64)
    g = _lowpass(a, alpha=0.15)                      # gravity estimate
    gn = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
    ghat = g / gn
    a_par = (a * ghat).sum(axis=-1, keepdims=True)   # projection along gravity
    a_perp = np.linalg.norm(a - a_par * ghat, axis=-1, keepdims=True)
    return a_par, a_perp, gn


def _rotation_invariant_stats(a):
    """a: (..., T, 3) -> rotation-invariant per-sample stats (..., T, 6)."""
    x = a[..., 0]; y = a[..., 1]; z = a[..., 2]
    return np.stack([x * x + y * y, y * y + z * z, x * x + z * z,
                     x * y, x * z, y * z], axis=-1)


def extract_orientation_invariant(windows: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """Orientation-invariant features per sensor (vectorised).

    windows: (N,T,3) test or (N,T,4,3) train. Returns (N, F) of added features.
    Includes: magnitude (already partly in base, expanded here), gravity/body-frame
    decomposition, rotation-invariant products, covariance eigenvalues.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 4:                       # (N,T,4,3)
        N, T, S, A = w.shape
        cols = []
        for s in range(S):
            cols.append(_orientation_invariant_one(w[:, :, s], fs))
        return np.concatenate(cols, axis=-1)
    else:                                 # (N,T,3)
        return _orientation_invariant_one(w, fs)


def _orientation_invariant_one(a, fs=50.0):
    """a: (N,T,3) -> (N, F) orientation-invariant features for one sensor."""
    mag = np.linalg.norm(a, axis=-1)                       # (N,T)
    # magnitude derivatives
    dm = np.gradient(mag, axis=-1)
    d2m = np.gradient(dm, axis=-1)
    a_par, a_perp, gn = _gravity_body_decompose(a)         # each (N,T,1)
    ri = _rotation_invariant_stats(a)                      # (N,T,6)
    # covariance eigenvalues (per window) of raw 3 axes
    N = a.shape[0]
    eig = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        c = np.cov(a[i].T)                                  # 3x3
        ev = np.linalg.eigvalsh(c)
        eig[i] = ev[::-1]
    # stack per-sample feature streams, then time-features each
    streams = np.concatenate([
        mag[:, :, None], dm[:, :, None], d2m[:, :, None],
        a_par, a_perp, gn, ri], axis=-1)                    # (N,T,12)
    feats = []
    for k in range(streams.shape[-1]):
        feats.append(_time_features_vect(streams[:, :, k]))
    feats.append(eig)                                        # 3 eigenvalues
    return np.concatenate(feats, axis=-1)


def extract_acc_features_vect(windows: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """Vectorised feature extraction.

    windows: (N, T, 3) single-sensor (test) or (N, T, 4, 3) all-sensors (train).
    Returns (N, F).  For (N,T,4,3) we concatenate per-sensor features (each
    sensor = 3 axes + magnitude).
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 4:  # (N,T,4,3)
        N, T, S, A = w.shape
        cols = []
        for s in range(S):
            for a in range(A):
                cols.append(_axis_features_vect(w[:, :, s, a], fs))
                cols.append(_subwindow_features_vect(w[:, :, s, a]))
            mag = np.linalg.norm(w[:, :, s], axis=-1)
            cols.append(_axis_features_vect(mag, fs))
            cols.append(_subwindow_features_vect(mag))
        return np.concatenate(cols, axis=-1)
    else:  # (N,T,3)
        N, T, A = w.shape
        cols = []
        for a in range(A):
            cols.append(_axis_features_vect(w[:, :, a], fs))
            cols.append(_subwindow_features_vect(w[:, :, a]))
        mag = np.linalg.norm(w, axis=-1)
        cols.append(_axis_features_vect(mag, fs))
        cols.append(_subwindow_features_vect(mag))
        return np.concatenate(cols, axis=-1)


def extract_video_features_vect(videos: np.ndarray,
                                n_components: int = 64) -> np.ndarray:
    """videos: (N, 15, 768) -> (N, 2*n_components) pooled features."""
    v = np.asarray(videos, dtype=np.float64)
    mean = v.mean(axis=1)                # (N,768)
    std = v.std(axis=1)
    feats = np.concatenate([mean, std], axis=-1)  # (N,1536)
    rng = np.random.RandomState(0)
    proj = rng.randn(1536, n_components) / np.sqrt(1536)
    return feats @ proj