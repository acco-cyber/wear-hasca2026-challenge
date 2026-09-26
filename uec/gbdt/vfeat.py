"""Vectorised (batch) port of approach_base.src.train_inertial_gbdt.make_feature_vector.

The reference computes ~874 features per (50,3) window with ~1,500 small numpy calls (~70 ms/window here),
which is ~9 CPU-hours for the training table. This module computes the SAME features, in the SAME column
order and with the same float32/float64 arithmetic, for a whole batch of windows at once.
check_vfeat.py compares it against the reference implementation on random windows.
"""
import numpy as np

f32, f64 = np.float32, np.float64
IMU_HZ = 50.0
SENSOR_TO_ID = {"ra": 0, "la": 1, "rl": 2, "ll": 3}


class Feats:
    def __init__(self):
        self.names, self.cols = [], []

    def add(self, name, arr):
        self.names.append(name)
        self.cols.append(np.asarray(arr, dtype=f64))

    def update(self, pairs):
        for n, a in pairs:
            self.add(n, a)


# ---------------------------------------------------------------- scalar-series helpers, X is (N, L) float32
def _mean(X):
    return X.mean(1)


def _std(X):
    return X.std(1)


def _var(X):
    return X.var(1)


def _iqr(X):
    return np.quantile(X, 0.75, axis=1) - np.quantile(X, 0.25, axis=1)


def _shape_factor(X):
    rms = np.sqrt(np.mean(np.square(X), 1)).astype(f64)
    mean_abs = np.abs(X.mean(1).astype(f64))
    return rms / np.maximum(mean_abs, 1e-6)


def _skew_kurt(X):
    std = X.std(1)
    ok = std.astype(f64) >= 1e-6
    safe = np.where(ok, std, f32(1.0)).astype(f32)
    C = (X - X.mean(1)[:, None]) / safe[:, None]
    sk = np.mean(np.power(C, 3), 1)
    ku = np.mean(np.power(C, 4), 1)
    return np.where(ok, sk.astype(f64), 0.0), np.where(ok, ku.astype(f64), 0.0)


def _zero_crossings64(C):
    """C float64 (N,L) already centred. Sign changes between consecutive non-zero samples."""
    N, L = C.shape
    if L < 2:
        return np.zeros(N)
    S = np.sign(C)
    nz = S != 0
    idx = np.where(nz, np.arange(L)[None, :], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    Sff = np.take_along_axis(S, idx, 1)
    return ((Sff[:, 1:] * Sff[:, :-1]) < 0).sum(1).astype(f64)


def _zero_crossings(X):
    return _zero_crossings64(np.asarray(X, dtype=f64))


def _mean_crossing_rate(X):
    L = X.shape[1]
    if L < 2:
        return np.zeros(X.shape[0])
    c = _zero_crossings64(X.astype(f64) - X.mean(1).astype(f64)[:, None])
    return c / max(L - 1, 1)


def _hist_entropy(X, n_bins=10):
    """np.histogram(x, bins=10) per row (float32 fast path of numpy), then Shannon entropy."""
    N, L = X.shape
    first = X.min(1)
    last = X.max(1)
    eq = first == last
    first = np.where(eq, first - f32(0.5), first).astype(f32)
    last = np.where(eq, last + f32(0.5), last).astype(f32)
    delta = (last - first).astype(f32)
    step = (delta / f32(n_bins)).astype(f32)
    edges = np.arange(0, n_bins + 1, dtype=f32)[None, :] * step[:, None]
    edges = edges + first[:, None]
    edges[:, -1] = last
    edges = edges.astype(f32)
    fidx = ((X - first[:, None]) / delta[:, None]) * f32(n_bins)
    ind = fidx.astype(np.intp)
    ind[ind == n_bins] -= 1
    dec = X < np.take_along_axis(edges, ind, 1)
    ind[dec] -= 1
    inc = (X >= np.take_along_axis(edges, ind + 1, 1)) & (ind != n_bins - 1)
    ind[inc] += 1
    H = np.zeros((N, n_bins), dtype=f64)
    np.add.at(H, (np.repeat(np.arange(N), L), ind.ravel()), 1.0)
    tot = H.sum(1)
    P = H / np.where(tot > 0, tot, 1.0)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0)
    return np.where(tot > 0, -T.sum(1), 0.0)


def _diff_entropy(X):
    var = X.var(1).astype(f64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 0.5 * np.log(2.0 * np.pi * np.e * np.where(var <= 1e-12, 1.0, var))
    return np.where(var <= 1e-12, 0.0, v)


def _hjorth(X):
    L = X.shape[1]
    var0 = X.var(1).astype(f64)
    dX = np.diff(X, axis=1)
    var1 = dX.var(1).astype(f64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mob = np.sqrt(var1 / np.where(var0 <= 1e-12, 1.0, var0))
    mob = np.where((var0 <= 1e-12) | (L < 2), 0.0, mob)
    var2 = np.diff(dX, axis=1).var(1).astype(f64)
    with np.errstate(divide="ignore", invalid="ignore"):
        comp = np.sqrt(var2 / np.where(var1 <= 1e-12, 1.0, var1)) / np.where(mob <= 1e-12, 1.0, mob)
    comp = np.where((L < 3) | (mob <= 1e-12) | (var1 <= 1e-12), 0.0, comp)
    return mob, comp


def _petrosian(X):
    n = X.shape[1]
    nd = _zero_crossings(np.diff(X, axis=1))
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nd + 1e-12)))


def _katz(X):
    n = X.shape[1]
    length = np.sum(np.abs(np.diff(X, axis=1)), 1).astype(f64)
    dist = np.max(np.abs(X - X[:, :1]), 1).astype(f64)
    bad = (length <= 1e-12) | (dist <= 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log10(n) / (np.log10(np.where(bad, 1.0, dist) / np.where(bad, 1.0, length)) + np.log10(n))
    return np.where(bad, 0.0, v)


def _spectral_entropy64(P):
    """P float64 (N,K) power."""
    tot = P.sum(1)
    Q = P / np.where(tot > 0, tot, 1.0)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(Q > 0, Q * np.log(np.where(Q > 0, Q, 1.0)), 0.0)
    return np.where(tot > 0, -T.sum(1), 0.0)


def _welch_entropy(X):
    x = X.astype(f64)
    n = x.shape[1]
    if n < 8:
        return np.zeros(x.shape[0])
    seg_len = max(n // 2, 8)
    step = max(seg_len // 2, 1)
    win = np.hanning(seg_len)
    psd = None
    count = 0
    for s in range(0, n - seg_len + 1, step):
        seg = x[:, s:s + seg_len]
        seg = (seg - np.mean(seg, 1)[:, None]) * win[None, :]
        p = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        psd = p if psd is None else psd + p
        count += 1
    psd = psd / float(count)
    psd = psd[:, 1:]
    return _spectral_entropy64(psd)


PEAK_NAMES = ["peak_count", "peak_mean", "peak_std", "peak_max", "peak_min", "peak_interval_mean",
              "peak_interval_std", "peak_first_pos", "peak_last_pos", "peak_prominence_mean",
              "peak_prominence_std", "peak_prominence_max"]


def _masked_mean_std(V, M):
    """float64 mean / std of V over mask M per row (0 where empty)."""
    cnt = M.sum(1)
    c = np.where(cnt > 0, cnt, 1)
    s = np.where(M, V, 0.0).sum(1)
    mu = s / c
    var = np.where(M, (V - mu[:, None]) ** 2, 0.0).sum(1) / c
    return np.where(cnt > 0, mu, 0.0), np.where(cnt > 0, np.sqrt(var), 0.0), cnt


def _peak_stats(X):
    N, L = X.shape
    out = {k: np.zeros(N) for k in PEAK_NAMES}
    if L < 3:
        return out
    M = (X[:, 1:-1] > X[:, :-2]) & (X[:, 1:-1] >= X[:, 2:])   # positions 1..L-2
    V = X[:, 1:-1]
    cnt = M.sum(1)
    has = cnt > 0
    V64 = V.astype(f64)
    # float32 statistics of the peak values (rounded like the float32 reference)
    mu, sd, _ = _masked_mean_std(V64, M)
    out["peak_count"] = cnt.astype(f64)
    out["peak_mean"] = np.where(has, mu.astype(f32).astype(f64), 0.0)
    out["peak_std"] = np.where(has, sd.astype(f32).astype(f64), 0.0)
    out["peak_max"] = np.where(has, np.where(M, V, -np.inf).max(1).astype(f64), 0.0)
    out["peak_min"] = np.where(has, np.where(M, V, np.inf).min(1).astype(f64), 0.0)
    pos = np.arange(1, L - 1)[None, :]
    P = np.sort(np.where(M, pos, 10 ** 6), axis=1)
    I = np.diff(P, axis=1).astype(f64)
    IM = np.arange(L - 3)[None, :] < (cnt - 1)[:, None]
    imu, isd, icnt = _masked_mean_std(I, IM)
    out["peak_interval_mean"] = np.where(icnt > 0, imu, 0.0)
    out["peak_interval_std"] = np.where(icnt > 0, isd, 0.0)
    denom = float(max(L - 1, 1))
    out["peak_first_pos"] = np.where(has, P[:, 0] / denom, 0.0)
    lastp = np.where(M, pos, -1).max(1)
    out["peak_last_pos"] = np.where(has, lastp / denom, 0.0)
    prom = (X[:, 1:-1] - np.maximum(X[:, :-2], X[:, 2:])).astype(f32)
    pmu, psd, _ = _masked_mean_std(prom.astype(f64), M)
    out["peak_prominence_mean"] = np.where(has, pmu.astype(f32).astype(f64), 0.0)
    out["peak_prominence_std"] = np.where(has, psd.astype(f32).astype(f64), 0.0)
    out["peak_prominence_max"] = np.where(has, np.where(M, prom, -np.inf).max(1).astype(f64), 0.0)
    return out


def _orientation(mx, my, mz, prefix=""):
    """mx,my,mz float64 arrays holding float32 means."""
    p = f"{prefix}_" if prefix else ""
    denom = np.sqrt(mx * mx + my * my + mz * mz)
    small = denom <= 1e-6
    d = np.where(small, 1.0, denom)
    tx = np.where(small, 0.0, np.arccos(np.clip(mx / d, -1.0, 1.0)))
    ty = np.where(small, 0.0, np.arccos(np.clip(my / d, -1.0, 1.0)))
    tz = np.where(small, 0.0, np.arccos(np.clip(mz / d, -1.0, 1.0)))
    return {
        f"{p}tilt_to_x": tx,
        f"{p}tilt_to_y": ty,
        f"{p}tilt_to_z": tz,
        f"{p}pitch": np.arctan2(-mx, np.sqrt(my * my + mz * mz)),
        f"{p}roll": np.arctan2(my, mz),
    }


def _fft_stats(X, sr=IMU_HZ):
    C = X - X.mean(1)[:, None]
    spec = np.fft.rfft(C, axis=1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / sr)
    power = power[:, 1:]
    freqs = freqs[1:]
    low = power[:, (freqs >= 0.0) & (freqs < 3.0)].sum(1)
    mid = power[:, (freqs >= 3.0) & (freqs < 10.0)].sum(1)
    high = power[:, freqs >= 10.0].sum(1)
    dom = np.argmax(power, 1)
    return [("fft_bandpower_low", low), ("fft_bandpower_mid", mid), ("fft_bandpower_high", high),
            ("fft_dom_freq", freqs[dom]), ("fft_dom_power", np.take_along_axis(power, dom[:, None], 1)[:, 0]),
            ("fft_entropy", _spectral_entropy64(power.astype(f64)))]


def _moving_average(X, w):
    w = max(int(w), 1)
    if w <= 1:
        return X.astype(f32)
    kern = (np.ones(w, dtype=f32) / float(w)).astype(f32)
    N, L = X.shape
    # np.convolve(x, kern, mode="same") for an odd symmetric kernel
    half = (w - 1) // 2
    pad = np.zeros((N, L + w - 1), dtype=f32)
    pad[:, half:half + L] = X
    out = np.zeros((N, L), dtype=f32)
    for j in range(w):
        out += pad[:, j:j + L] * kern[w - 1 - j]
    return out


def _segment_stats(X, prefix, n):
    res = []
    for i, seg in enumerate(np.array_split(X, n, axis=1), start=1):
        if seg.shape[1] == 0:
            z = np.zeros(X.shape[0])
            vals = (z, z, z, z)
        else:
            vals = (seg.mean(1), seg.std(1), seg.min(1), seg.max(1))
        for nm, v in zip(("mean", "std", "min", "max"), vals):
            res.append((f"{prefix}_seg{n}_{i}_{nm}", v))
    return res


def _axis_energy_ratio(W):
    """W (N,L,3) float32 -> dict of float32-division ratios."""
    ae = np.mean(W * W, axis=1)                    # (N,3) float32
    tot = ae.sum(1)
    small = tot.astype(f64) <= 1e-12
    safe = np.where(small, f32(1.0), tot).astype(f32)
    r = ae / safe[:, None]
    return {"energy_ratio_x": np.where(small, 0.0, r[:, 0].astype(f64)),
            "energy_ratio_y": np.where(small, 0.0, r[:, 1].astype(f64)),
            "energy_ratio_z": np.where(small, 0.0, r[:, 2].astype(f64))}


def _cov3(W):
    """np.cov(window, rowvar=False) per row, float64 (N,3,3)."""
    X = W.astype(f64)
    X = X - X.mean(1, keepdims=True)
    n = W.shape[1]
    c = np.einsum("nti,ntj->nij", X, X)
    return c * np.true_divide(1, n - 1)


def _corr_from(W, c, i, j):
    sa = W[:, :, i].std(1).astype(f64)
    sb = W[:, :, j].std(1).astype(f64)
    zero = (sa == 0.0) | (sb == 0.0)
    si = np.sqrt(c[:, i, i])
    sj = np.sqrt(c[:, j, j])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (c[:, i, j] / si) / sj
    r = np.clip(r, -1.0, 1.0)
    return np.where(zero, 0.0, r)


def _multiaxis_segment(W, prefix, n):
    res = []
    for i, seg in enumerate(np.array_split(W, n, axis=1), start=1):
        sp = f"{prefix}_seg{n}_{i}"
        if seg.shape[1] < 2:
            z = np.zeros(W.shape[0])
            for nm in ("corr_xy", "corr_xz", "corr_yz", "cov_xx", "cov_yy", "cov_zz", "cov_xy", "cov_xz", "cov_yz"):
                res.append((f"{sp}_{nm}", z))
            for nm in ("energy_ratio_x", "energy_ratio_y", "energy_ratio_z"):
                res.append((f"{sp}_{nm}", z))
            continue
        c = _cov3(seg)
        res.append((f"{sp}_corr_xy", _corr_from(seg, c, 0, 1)))
        res.append((f"{sp}_corr_xz", _corr_from(seg, c, 0, 2)))
        res.append((f"{sp}_corr_yz", _corr_from(seg, c, 1, 2)))
        res.append((f"{sp}_cov_xx", c[:, 0, 0]))
        res.append((f"{sp}_cov_yy", c[:, 1, 1]))
        res.append((f"{sp}_cov_zz", c[:, 2, 2]))
        res.append((f"{sp}_cov_xy", c[:, 0, 1]))
        res.append((f"{sp}_cov_xz", c[:, 0, 2]))
        res.append((f"{sp}_cov_yz", c[:, 1, 2]))
        for nm, v in _axis_energy_ratio(seg).items():
            res.append((f"{sp}_{nm}", v))
    return res


def _vnorm(M):
    """np.linalg.norm of float32 3-vectors, row-wise."""
    return np.sqrt((M * M).sum(-1))


def _edge_mean_diff(W, prefix, n):
    N = W.shape[0]
    if W.shape[1] < n * 2:
        z = np.zeros(N)
        names = ["mean_diff_x", "mean_diff_y", "mean_diff_z", "mean_diff_norm", "euclidean_mean_diff", "pitch_diff",
                 "roll_diff", "tilt_to_x_diff", "tilt_to_y_diff", "tilt_to_z_diff", "energy_ratio_x_diff",
                 "energy_ratio_y_diff", "energy_ratio_z_diff"]
        return [(f"{prefix}_edge{n}_{k}", z) for k in names]
    first = W[:, :n]
    last = W[:, -n:]
    fm = first.mean(1)
    lm = last.mean(1)
    d = lm - fm
    res = [(f"{prefix}_edge{n}_mean_diff_x", d[:, 0]), (f"{prefix}_edge{n}_mean_diff_y", d[:, 1]),
           (f"{prefix}_edge{n}_mean_diff_z", d[:, 2]),
           (f"{prefix}_edge{n}_mean_diff_norm", _vnorm(lm) - _vnorm(fm)),
           (f"{prefix}_edge{n}_euclidean_mean_diff", _vnorm(d))]
    fo = _orientation(*[fm[:, k].astype(f64) for k in range(3)], prefix="first")
    lo = _orientation(*[lm[:, k].astype(f64) for k in range(3)], prefix="last")
    for nm in ("pitch", "roll", "tilt_to_x", "tilt_to_y", "tilt_to_z"):
        res.append((f"{prefix}_edge{n}_{nm}_diff", np.abs(lo[f"last_{nm}"] - fo[f"first_{nm}"])))
    fe = _axis_energy_ratio(first)
    le = _axis_energy_ratio(last)
    for nm in ("energy_ratio_x", "energy_ratio_y", "energy_ratio_z"):
        res.append((f"{prefix}_edge{n}_{nm}_diff", np.abs(le[nm] - fe[nm])))
    return res


def _scalar_edge_diff(X, prefix, n):
    N = X.shape[0]
    if X.shape[1] < n * 2:
        z = np.zeros(N)
        return [(f"{prefix}_edge{n}_{k}", z) for k in ("mean_diff", "abs_mean_diff", "std_diff", "rms_diff")]
    first = X[:, :n]
    last = X[:, -n:]
    md = (last.mean(1) - first.mean(1)).astype(f64)
    frms = np.sqrt(np.mean(first * first, 1)).astype(f64)
    lrms = np.sqrt(np.mean(last * last, 1)).astype(f64)
    return [(f"{prefix}_edge{n}_mean_diff", md), (f"{prefix}_edge{n}_abs_mean_diff", np.abs(md)),
            (f"{prefix}_edge{n}_std_diff", last.std(1) - first.std(1)), (f"{prefix}_edge{n}_rms_diff", lrms - frms)]


def _smoothed_diff(X, prefix, w):
    S = _moving_average(X, w)
    D = np.diff(S, axis=1)
    G = np.gradient(S, axis=1)
    return [(f"{prefix}_smdiff_mean", D.mean(1)), (f"{prefix}_smdiff_std", D.std(1)),
            (f"{prefix}_smdiff_abs_mean", np.abs(D).mean(1)), (f"{prefix}_smdiff_rms", np.sqrt(np.mean(D * D, 1))),
            (f"{prefix}_cdiff_mean", G.mean(1)), (f"{prefix}_cdiff_std", G.std(1)),
            (f"{prefix}_cdiff_abs_mean", np.abs(G).mean(1)), (f"{prefix}_cdiff_rms", np.sqrt(np.mean(G * G, 1)))]


def _run_length(M):
    N, L = M.shape
    starts = M.copy()
    starts[:, 1:] &= ~M[:, :-1]
    nrun = starts.sum(1).astype(f64)
    tot = M.sum(1).astype(f64)
    run = np.zeros(N, dtype=np.int64)
    mx = np.zeros(N, dtype=np.int64)
    for t in range(L):
        run = (run + 1) * M[:, t]
        mx = np.maximum(mx, run)
    mean = np.where(nrun > 0, tot / np.where(nrun > 0, nrun, 1.0), 0.0)
    return nrun, mean, mx.astype(f64)


LAG_KEYS = ["diff_mean", "diff_std", "diff_min", "diff_max", "diff_rms", "diff_abs_mean", "diff_skewness",
            "diff_kurtosis", "diff_zero_crossings", "diff_mean_crossing_rate", "diff_sign_change_count",
            "diff_sign_change_rate", "diff_pos_run_count", "diff_pos_run_mean", "diff_pos_run_max",
            "diff_neg_run_count", "diff_neg_run_mean", "diff_neg_run_max", "diff_pos_ratio", "diff_neg_ratio",
            "diff_zero_ratio", "diff_pos_mean", "diff_neg_mean"]


def _lag_diff(X, prefix, lag):
    N, L = X.shape
    if lag <= 0 or L <= lag:
        z = np.zeros(N)
        return [(f"{prefix}_lag{lag}_{k}", z) for k in LAG_KEYS]
    D = X[:, lag:] - X[:, :-lag]
    pos = D > 0
    neg = D < 0
    zero = D == 0
    D64 = D.astype(f64)
    pm, _, pc = _masked_mean_std(D64, pos)
    nm_, _, ncnt = _masked_mean_std(D64, neg)
    pos_mean = np.where(pc > 0, pm.astype(f32).astype(f64), 0.0)
    neg_mean = np.where(ncnt > 0, nm_.astype(f32).astype(f64), 0.0)
    zc = _zero_crossings(D)
    nnz = (np.sign(D64) != 0).sum(1)
    sc_rate = np.where(nnz >= 2, zc / np.where(nnz >= 2, nnz - 1, 1), 0.0)
    sc_count = np.where(nnz >= 2, zc, 0.0)
    sk, ku = _skew_kurt(D)
    prc, prm, prx = _run_length(pos)
    nrc, nrm, nrx = _run_length(neg)
    vals = [D.mean(1), D.std(1), D.min(1), D.max(1), np.sqrt(np.mean(D * D, 1)), np.abs(D).mean(1), sk, ku, zc,
            _mean_crossing_rate(D), sc_count, sc_rate, prc, prm, prx, nrc, nrm, nrx,
            pos.mean(1), neg.mean(1), zero.mean(1), pos_mean, neg_mean]
    return [(f"{prefix}_lag{lag}_{k}", v) for k, v in zip(LAG_KEYS, vals)]


def _series_block(F, x, name, smoothing_window):
    F.update(_smoothed_diff(x, name, smoothing_window))
    for lag in (5, 10, 20, 30):
        F.update(_lag_diff(x, name, lag))
    for n in (10, 20):
        F.update(_scalar_edge_diff(x, name, n))
    F.update(_segment_stats(x, name, 3))
    F.update(_segment_stats(x, name, 5))
    for k, v in _peak_stats(x).items():
        F.add(f"{name}_{k}", v)
    for k, v in _fft_stats(x):
        F.add(f"{name}_{k}", v)


def make_feature_matrix(W, sensor_keys, smoothing_window=5, sensor_embedding_mode="sensor"):
    """W (N,50,3) windows (any float dtype); sensor_keys: str or array of N sensor keys.
    Returns (names, X float64 (N,F)) matching make_feature_vector(use_raw_features=False)."""
    W = np.nan_to_num(np.asarray(W, dtype=f32), nan=0.0, posinf=0.0, neginf=0.0).astype(f32)
    N = W.shape[0]
    F = Feats()
    axis_means = []
    for a, an in enumerate("xyz"):
        x = np.ascontiguousarray(W[:, :, a])
        dx = np.diff(x, axis=1)
        ddx = np.diff(dx, axis=1)
        half = x.shape[1] // 2
        m = x.mean(1)
        axis_means.append(m)
        F.add(f"{an}_mean", m)
        F.add(f"{an}_variance", x.var(1))
        F.add(f"{an}_std", x.std(1))
        F.add(f"{an}_min", x.min(1))
        F.add(f"{an}_max", x.max(1))
        F.add(f"{an}_median", np.median(x, axis=1))
        F.add(f"{an}_q25", np.quantile(x, 0.25, axis=1))
        F.add(f"{an}_q75", np.quantile(x, 0.75, axis=1))
        F.add(f"{an}_iqr", _iqr(x))
        F.add(f"{an}_range", x.max(1) - x.min(1))
        F.add(f"{an}_rms", np.sqrt(np.mean(x * x, 1)))
        F.add(f"{an}_shape_factor", _shape_factor(x))
        F.add(f"{an}_abs_mean", np.abs(x).mean(1))
        sk, ku = _skew_kurt(x)
        F.add(f"{an}_skewness", sk)
        F.add(f"{an}_kurtosis", ku)
        F.add(f"{an}_zero_crossings", _zero_crossings(x))
        F.add(f"{an}_mean_crossing_rate", _mean_crossing_rate(x))
        F.add(f"{an}_signal_entropy", _hist_entropy(x))
        F.add(f"{an}_diff_entropy", _diff_entropy(x))
        mob, comp = _hjorth(x)
        F.add(f"{an}_hjorth_mobility", mob)
        F.add(f"{an}_hjorth_complexity", comp)
        F.add(f"{an}_petrosian_fd", _petrosian(x))
        F.add(f"{an}_katz_fd", _katz(x))
        F.add(f"{an}_diff_mean", dx.mean(1))
        F.add(f"{an}_diff_std", dx.std(1))
        F.add(f"{an}_diff_abs_mean", np.abs(dx).mean(1))
        F.add(f"{an}_jerk_mean", ddx.mean(1))
        F.add(f"{an}_jerk_std", ddx.std(1))
        F.add(f"{an}_jerk_abs_mean", np.abs(ddx).mean(1))
        F.add(f"{an}_half_mean_diff", np.abs(x[:, half:].mean(1) - x[:, :half].mean(1)))
        _series_block(F, x, an, smoothing_window)

    norm = np.sqrt((W * W).sum(-1))                       # (N,50) float32
    norm_l1 = np.abs(W).sum(-1)
    dnorm = np.diff(norm, axis=1)
    ddnorm = np.diff(dnorm, axis=1)
    dW = np.diff(W, axis=1)
    diff_norm = np.sqrt((dW * dW).sum(-1))
    jW = np.diff(dW, axis=1)
    jerk_norm = np.sqrt((jW * jW).sum(-1))
    half = norm.shape[1] // 2
    am64 = [m.astype(f64) for m in axis_means]
    F.add("sma", norm_l1.mean(1))
    F.add("total_mean_signed", am64[0] + am64[1] + am64[2])
    F.add("total_mean_euclidean", np.sqrt(np.square(am64[0]) + np.square(am64[1]) + np.square(am64[2])))
    F.add("norm_mean", norm.mean(1))
    F.add("norm_variance", norm.var(1))
    F.add("norm_std", norm.std(1))
    F.add("norm_min", norm.min(1))
    F.add("norm_max", norm.max(1))
    F.add("norm_peak_to_peak", norm.max(1) - norm.min(1))
    F.add("norm_iqr", _iqr(norm))
    F.add("norm_rms", np.sqrt(np.mean(norm * norm, 1)))
    F.add("norm_l1_mean", norm_l1.mean(1))
    F.add("norm_l1_std", norm_l1.std(1))
    sk, ku = _skew_kurt(norm)
    F.add("norm_skewness", sk)
    F.add("norm_kurtosis", ku)
    F.add("norm_diff_abs_mean", np.abs(dnorm).mean(1))
    F.add("diff_norm_mean", diff_norm.mean(1))
    F.add("diff_norm_std", diff_norm.std(1))
    F.add("diff_norm_min", diff_norm.min(1))
    F.add("diff_norm_max", diff_norm.max(1))
    F.add("diff_norm_rms", np.sqrt(np.mean(diff_norm * diff_norm, 1)))
    F.add("diff_norm_abs_mean", np.abs(diff_norm).mean(1))
    F.add("diff_norm_zero_crossings", _zero_crossings(dnorm))
    F.add("jerk_norm_mean", jerk_norm.mean(1))
    F.add("jerk_norm_std", jerk_norm.std(1))
    F.add("jerk_norm_min", jerk_norm.min(1))
    F.add("jerk_norm_max", jerk_norm.max(1))
    F.add("jerk_norm_rms", np.sqrt(np.mean(jerk_norm * jerk_norm, 1)))
    F.add("jerk_norm_abs_mean", np.abs(jerk_norm).mean(1))
    F.add("norm_jerk_abs_mean", np.abs(ddnorm).mean(1))
    F.add("norm_half_mean_diff", np.abs(norm[:, half:].mean(1) - norm[:, :half].mean(1)))
    F.add("norm_mean_crossing_rate", _mean_crossing_rate(norm))
    F.add("norm_signal_entropy", _hist_entropy(norm))
    F.add("norm_diff_entropy", _diff_entropy(norm))
    mob, comp = _hjorth(norm)
    F.add("norm_hjorth_mobility", mob)
    F.add("norm_hjorth_complexity", comp)
    F.add("norm_petrosian_fd", _petrosian(norm))
    F.add("norm_katz_fd", _katz(norm))
    F.add("norm_welch_entropy", _welch_entropy(norm))
    _series_block(F, norm, "norm", smoothing_window)

    ori = _orientation(*am64)
    for k, v in ori.items():
        F.add(k, v)
    F.add("tilt_angle", ori["tilt_to_z"])
    for k, v in _axis_energy_ratio(W).items():
        F.add(k, v)
    for n in (10, 20):
        F.update(_edge_mean_diff(W, "window", n))
    fh = W[:, :half].mean(1)
    sh = W[:, half:].mean(1)
    fo = _orientation(*[fh[:, k].astype(f64) for k in range(3)], prefix="first_half")
    so = _orientation(*[sh[:, k].astype(f64) for k in range(3)], prefix="second_half")
    for nm in ("pitch", "roll", "tilt_to_x", "tilt_to_y", "tilt_to_z"):
        F.add(f"{nm}_half_diff", np.abs(so[f"second_half_{nm}"] - fo[f"first_half_{nm}"]))
    for i, seg in enumerate(np.array_split(W, 3, axis=1), start=1):
        sm = seg.mean(1)
        for k, v in _orientation(*[sm[:, j].astype(f64) for j in range(3)], prefix=f"seg3_{i}").items():
            F.add(k, v)
    F.update(_multiaxis_segment(W, "window", 3))
    c = _cov3(W)
    F.add("corr_xy", _corr_from(W, c, 0, 1))
    F.add("corr_xz", _corr_from(W, c, 0, 2))
    F.add("corr_yz", _corr_from(W, c, 1, 2))
    F.add("cov_xx", c[:, 0, 0])
    F.add("cov_yy", c[:, 1, 1])
    F.add("cov_zz", c[:, 2, 2])
    F.add("cov_xy", c[:, 0, 1])
    F.add("cov_xz", c[:, 0, 2])
    F.add("cov_yz", c[:, 1, 2])
    ev = np.linalg.eigvalsh(c)
    ev = np.sort(np.clip(ev, 0.0, None), axis=1)[:, ::-1]
    tot = ev.sum(1)
    F.add("pca_eig1", ev[:, 0])
    F.add("pca_eig2", ev[:, 1])
    F.add("pca_eig3", ev[:, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        F.add("pca_ratio_12", np.where(ev[:, 1] > 0.0, ev[:, 0] / np.where(ev[:, 1] > 0, ev[:, 1], 1.0), 0.0))
        F.add("pca_ratio_13", np.where(ev[:, 2] > 0.0, ev[:, 0] / np.where(ev[:, 2] > 0, ev[:, 2], 1.0), 0.0))
        tt = np.where(tot > 0, tot, 1.0)
        F.add("pca_energy_1", np.where(tot > 0.0, ev[:, 0] / tt, 0.0))
        F.add("pca_energy_2", np.where(tot > 0.0, ev[:, 1] / tt, 0.0))
        F.add("pca_energy_3", np.where(tot > 0.0, ev[:, 2] / tt, 0.0))

    if sensor_embedding_mode == "sensor":
        keys = np.broadcast_to(np.asarray(sensor_keys), (N,))
        sid = np.array([SENSOR_TO_ID[k] for k in keys], dtype=f64)
        F.add("sensor_id", sid)
        for k, v in SENSOR_TO_ID.items():
            F.add(f"sensor_is_{k}", (sid == v).astype(f64))
    elif sensor_embedding_mode != "none":
        raise ValueError(sensor_embedding_mode)
    return F.names, np.stack(F.cols, axis=1)
