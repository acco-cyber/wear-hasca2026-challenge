"""Phase-free per-window descriptors (the four sensors are not sample-synchronised, offsets wander up to +-1.5 s,
so lag-0 cross-correlation carries no signal; only activity-signature features can match limbs)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from scipy.stats import skew, kurtosis

WIN = 50
BANDS = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13), (13, 26)]  # rfft bins == Hz for a 50-sample 50 Hz window
DESC_NAMES = (["logstd", "logjerk", "logrange", "meanmag", "minmag", "maxmag", "skew", "kurt",
               "gx", "gy", "gz", "gnorm", "lsx", "lsy", "lsz", "vshare",
               "fdom", "centroid", "sentropy", "logpow"]
              + [f"band{i}" for i in range(len(BANDS))]
              + [f"acf{l}" for l in (3, 6, 10, 15, 20, 25)]
              + ["lsx_hp", "lsy_hp", "lsz_hp", "domaxis"])
ND = len(DESC_NAMES)


def fill_nan(w):
    w = w.copy()
    bad = np.isnan(w)
    if not bad.any():
        return w
    idx = np.where(bad.any(axis=(1, 2)))[0]
    t = np.arange(w.shape[1])
    for i in idx:
        for c in range(3):
            m = bad[i, :, c]
            if m.all():
                w[i, :, c] = 0.0
            elif m.any():
                w[i, m, c] = np.interp(t[m], t[~m], w[i, ~m, c])
    return w


def _zn(x, eps=1e-6):
    x = x - x.mean(axis=1, keepdims=True)
    s = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return (x / (s + eps)).astype(np.float32)


def describe(w):
    """w: (N,50,3) -> dict(desc (N,ND) float32, z_spec (N,25), z_specj (N,25), z_acf (N,25), std (N,), dyn (N,) bool)."""
    w = fill_nan(np.asarray(w, dtype=np.float32)).astype(np.float64)
    N = w.shape[0]
    mag = np.sqrt((w ** 2).sum(-1))
    hp = mag - mag.mean(1, keepdims=True)
    std = mag.std(1)
    dmag = np.diff(mag, axis=1)
    jerk = np.sqrt((np.diff(w, axis=1) ** 2).sum(-1))
    gmean = w.mean(1)
    gnorm = np.linalg.norm(gmean, axis=1)
    ghat = gmean / (gnorm[:, None] + 1e-6)
    vert = (w * ghat[:, None, :]).sum(-1)
    vert_hp = vert - vert.mean(1, keepdims=True)
    tot_var = ((w - gmean[:, None, :]) ** 2).sum((1, 2)) + 1e-9
    vshare = (vert_hp ** 2).sum(1) / tot_var
    F = np.fft.rfft(hp, axis=1)
    pw = np.abs(F[:, 1:26]) ** 2  # bins 1..25 Hz
    ptot = pw.sum(1) + 1e-9
    pn = pw / ptot[:, None]
    freqs = np.arange(1, 26)
    fdom = freqs[np.argmax(pw, axis=1)]
    centroid = (pn * freqs).sum(1)
    sentropy = -(pn * np.log(pn + 1e-12)).sum(1)
    bands = np.stack([np.log(pn[:, lo - 1:hi - 1].sum(1) + 1e-6) for lo, hi in BANDS], 1)
    Fj = np.fft.rfft(jerk - jerk.mean(1, keepdims=True), n=50, axis=1)
    pwj = np.abs(Fj[:, 1:26]) ** 2
    # autocorrelation of hp magnitude (biased), lags 1..25
    acf_full = np.fft.irfft(np.abs(np.fft.rfft(hp, n=100, axis=1)) ** 2, n=100, axis=1)[:, :26]
    acf = acf_full[:, 1:] / (acf_full[:, :1] + 1e-9)
    hp3 = w - gmean[:, None, :]
    ls_hp = np.log(hp3.std(1) + 1e-4)
    domaxis = np.argmax(hp3.std(1), axis=1).astype(np.float64)
    desc = np.column_stack([
        np.log(std + 1e-4), np.log(jerk.mean(1) + 1e-4), np.log(mag.max(1) - mag.min(1) + 1e-4),
        mag.mean(1), mag.min(1), mag.max(1), skew(hp, axis=1), kurtosis(hp, axis=1),
        gmean, gnorm, np.log(w.std(1) + 1e-4), vshare,
        fdom, centroid, sentropy, np.log(ptot),
        bands,
        acf[:, [2, 5, 9, 14, 19, 24]],
        ls_hp, domaxis,
    ]).astype(np.float32)
    assert desc.shape[1] == ND, (desc.shape, ND)
    spec = np.log1p(pn * 25.0)
    specj = np.log1p(pwj / (pwj.sum(1, keepdims=True) + 1e-9) * 25.0)
    return dict(desc=desc, z_spec=_zn(spec), z_specj=_zn(specj), z_acf=_zn(acf), std=std.astype(np.float32),
                dyn=std > 0.08, gmean=gmean.astype(np.float32))


# ---------------- pair features ----------------
# desc_a (ND) + desc_b (ND) + |desc_a-desc_b| (ND) + c_spec, c_specj, c_acf + pair
PAIR_NAMES = [f"a_{n}" for n in DESC_NAMES] + [f"b_{n}" for n in DESC_NAMES] + [f"d_{n}" for n in DESC_NAMES] + \
             ["c_spec", "c_specj", "c_acf", "pair"]
NPF = len(PAIR_NAMES)


def pair_rows(da, db, ia, ib, pair_id, M=None):
    """Feature rows for pairs (ia[k], ib[k]). M: optional dict of precomputed full matrices c_spec/c_specj/c_acf."""
    A = da["desc"][ia]; B = db["desc"][ib]
    out = np.empty((len(ia), NPF), dtype=np.float32)
    out[:, :ND] = A; out[:, ND:2 * ND] = B; out[:, 2 * ND:3 * ND] = np.abs(A - B)
    k = 3 * ND
    if M is None:
        out[:, k] = (da["z_spec"][ia] * db["z_spec"][ib]).sum(1)
        out[:, k + 1] = (da["z_specj"][ia] * db["z_specj"][ib]).sum(1)
        out[:, k + 2] = (da["z_acf"][ia] * db["z_acf"][ib]).sum(1)
    else:
        out[:, k] = M["c_spec"][ia, ib]; out[:, k + 1] = M["c_specj"][ia, ib]; out[:, k + 2] = M["c_acf"][ia, ib]
    out[:, k + 3] = pair_id
    return out


def corr_matrices(da, db):
    return {"c_spec": da["z_spec"] @ db["z_spec"].T, "c_specj": da["z_specj"] @ db["z_specj"].T,
            "c_acf": da["z_acf"] @ db["z_acf"].T}


NUM_IT = int(os.environ.get("PAIR_NUM_IT", "150"))  # trees used at prediction (speed/quality trade-off)


def score_matrix(model, da, db, pair_id, chunk=2_000_000):
    """Dense N_a x N_b log-odds matrix from the LightGBM pair model (chunked)."""
    M = corr_matrices(da, db)
    Na, Nb = M["c_spec"].shape
    S = np.empty((Na, Nb), dtype=np.float32)
    rows_per = max(1, chunk // Nb)
    kw = {"num_iteration": NUM_IT} if hasattr(model, "num_trees") else {}
    for r0 in range(0, Na, rows_per):
        r1 = min(Na, r0 + rows_per)
        ia = np.repeat(np.arange(r0, r1), Nb); ib = np.tile(np.arange(Nb), r1 - r0)
        X = pair_rows(da, db, ia, ib, pair_id, M=M)
        S[r0:r1] = model.predict(X, raw_score=True, num_threads=6, **kw).reshape(r1 - r0, Nb)
    return S


def score_rows(model, da, db, ia, ib, pair_ids, chunk=2_000_000):
    """Log-odds for arbitrary pairs (ia[k], ib[k]) with per-row pair id (array)."""
    out = np.empty(len(ia), dtype=np.float32)
    for r0 in range(0, len(ia), chunk):
        r1 = min(len(ia), r0 + chunk)
        X = pair_rows(da, db, ia[r0:r1], ib[r0:r1], 0)
        X[:, -1] = pair_ids[r0:r1]
        out[r0:r1] = model.predict(X, raw_score=True, num_threads=6)
    return out
