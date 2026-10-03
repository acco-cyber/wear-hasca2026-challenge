"""Shared utilities: load train sessions, cut 1-s windows, per-window descriptors,
pairwise same-second feature matrices, LightGBM pair model, assignment and evaluation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

TRAIN_DIR = r"E:\Claude code\wear\data\train\inertial_feat"
HERE = os.path.dirname(os.path.abspath(__file__))
# our limb order: 0=left_arm,1=left_leg,2=right_arm,3=right_leg
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_ID = {p: i for i, p in enumerate(PAIRS)}
DYN_THR = 0.08
WIN = 50


def load_session(name):
    """Return acc (4, nsec, 50, 3) float32 and labels (nsec,) object (str, 'null' for NaN)."""
    df = pd.read_csv(os.path.join(TRAIN_DIR, name + ".csv"))
    n = (len(df) // WIN) * WIN
    df = df.iloc[:n]
    cols = []
    for limb in LIMBS:
        cols.append([f"{limb}_acc_{ax}" for ax in "xyz"])
    acc = np.stack([df[c].to_numpy(dtype=np.float32) for c in cols], 0)  # (4, n, 3)
    acc = acc.reshape(4, n // WIN, WIN, 3)
    lab = df["label"].astype(object).where(df["label"].notna(), "null").to_numpy()
    lab = lab.reshape(n // WIN, WIN)
    # majority label per second
    out = np.empty(lab.shape[0], dtype=object)
    for i in range(lab.shape[0]):
        v, c = np.unique(lab[i], return_counts=True)
        out[i] = v[np.argmax(c)]
    return acc, out


def fill_nan(w):
    """w: (N,50,3). Linear-interpolate NaNs along time per channel; all-NaN -> 0."""
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


def _z(x, axis=-1, eps=1e-6):
    x = x - x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return x / (s + eps)


def _zn(x, eps=1e-6):
    """z-normalise rows and scale by 1/sqrt(L) so that A @ B.T is a Pearson correlation."""
    x = x - x.mean(axis=1, keepdims=True)
    s = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return (x / (s + eps)).astype(np.float32)


def window_desc(w):
    """w: (N,50,3) float32 -> dict of per-window descriptors (all float32)."""
    w = fill_nan(np.asarray(w, dtype=np.float32))
    N = w.shape[0]
    mag = np.sqrt((w ** 2).sum(-1))  # (N,50)
    hp = mag - mag.mean(1, keepdims=True)
    std = mag.std(1)
    dmag = np.diff(mag, axis=1)  # (N,49)
    jerk = np.sqrt((np.diff(w, axis=1) ** 2).sum(-1))  # (N,49)
    # energy envelope: 5-pt moving average of |dmag|
    k = np.ones(5, dtype=np.float32) / 5
    env = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, np.abs(dmag))
    gmean = w.mean(1)  # (N,3)
    gnorm = np.linalg.norm(gmean, axis=1, keepdims=True) + 1e-6
    ghat = gmean / gnorm
    vert = (w * ghat[:, None, :]).sum(-1)  # (N,50) projection on gravity dir
    vert = vert - vert.mean(1, keepdims=True)
    horiz = np.sqrt(np.maximum((w ** 2).sum(-1) - (vert + (w * ghat[:, None, :]).sum(-1).mean(1, keepdims=True)) ** 2, 0))
    horiz = horiz - horiz.mean(1, keepdims=True)
    # spectrum of hp magnitude (bins 1..25 -> 1..25 Hz)
    F = np.fft.rfft(hp, axis=1)
    pw = np.abs(F[:, 1:26]) ** 2
    spec = np.log1p(pw / (pw.sum(1, keepdims=True) + 1e-9) * 25.0)  # normalised log spectrum
    fdom = (np.argmax(pw, axis=1) + 1).astype(np.float32)  # Hz
    # phase at dominant frequency bin
    ph = np.angle(F[np.arange(N), np.argmax(pw, axis=1) + 1])
    # jerk spectrum
    Fj = np.fft.rfft(jerk - jerk.mean(1, keepdims=True), n=50, axis=1)
    pwj = np.abs(Fj[:, 1:26]) ** 2
    specj = np.log1p(pwj / (pwj.sum(1, keepdims=True) + 1e-9) * 25.0)
    gstd = w.std(1)  # (N,3)
    d = dict(
        z_hp=_zn(hp), z_d=_zn(dmag), z_jerk=_zn(jerk), z_env=_zn(env), z_vert=_zn(vert), z_horiz=_zn(horiz),
        z_spec=_zn(spec), z_specj=_zn(specj),
        std=std.astype(np.float32), logstd=np.log(std + 1e-4).astype(np.float32),
        fdom=fdom, phase=ph.astype(np.float32), gmean=gmean.astype(np.float32), gstd=gstd.astype(np.float32),
        jerk_mean=jerk.mean(1).astype(np.float32), hp_absmax=np.abs(hp).max(1).astype(np.float32),
        dyn=(std > DYN_THR),
    )
    # per-axis z-normalised series for a 3x3 axis cross-correlation block
    d["z_ax"] = np.stack([_zn(w[:, :, c] - 0) for c in range(3)], 1)  # (N,3,50)
    return d


PAIR_FEATS = ["c_hp", "c_d", "c_jerk", "c_env", "c_vert", "c_horiz", "c_spec", "c_specj", "c_hp_lag",
              "d_logstd", "ad_logstd", "d_fdom", "cos_dph", "ax_max", "ax_absmean",
              "std_a", "std_b", "fdom_a", "fdom_b", "jm_a", "jm_b",
              "ga_x", "ga_y", "ga_z", "gb_x", "gb_y", "gb_z", "gs_a", "gs_b", "pair"]


def pair_matrices(da, db):
    """Dense N_a x N_b matrices of the correlation-type pair features (float32)."""
    M = {}
    for k in ["hp", "d", "jerk", "env", "vert", "horiz", "spec", "specj"]:
        M["c_" + k] = da["z_" + k] @ db["z_" + k].T
    # lagged hp correlation (max over +-2 samples) to absorb small inter-sensor clock offsets
    best = M["c_hp"].copy()
    A = da["z_hp"]; B = db["z_hp"]
    for lag in (-2, -1, 1, 2):
        if lag > 0:
            c = A[:, lag:] @ B[:, :-lag].T
        else:
            c = A[:, :lag] @ B[:, -lag:].T
        np.maximum(best, c, out=best)
    M["c_hp_lag"] = best
    # axis-wise cross-correlation block: max and mean |.| over the 3x3 combos
    ax = np.zeros_like(M["c_hp"]); axabs = np.zeros_like(M["c_hp"])
    for i in range(3):
        for j in range(3):
            c = da["z_ax"][:, i] @ db["z_ax"][:, j].T
            np.maximum(ax, c, out=ax)
            axabs += np.abs(c)
    M["ax_max"] = ax; M["ax_absmean"] = axabs / 9.0
    return M


def pair_feature_rows(da, db, ia, ib, pair_id, M=None):
    """Feature rows for the pairs (ia[k], ib[k]). If M (full matrices) given, index them; else compute directly."""
    n = len(ia)
    out = np.empty((n, len(PAIR_FEATS)), dtype=np.float32)
    col = {f: i for i, f in enumerate(PAIR_FEATS)}
    if M is None:
        for k in ["hp", "d", "jerk", "env", "vert", "horiz", "spec", "specj"]:
            out[:, col["c_" + k]] = (da["z_" + k][ia] * db["z_" + k][ib]).sum(1)
        A = da["z_hp"][ia]; B = db["z_hp"][ib]
        best = (A * B).sum(1)
        for lag in (-2, -1, 1, 2):
            if lag > 0:
                c = (A[:, lag:] * B[:, :-lag]).sum(1)
            else:
                c = (A[:, :lag] * B[:, -lag:]).sum(1)
            best = np.maximum(best, c)
        out[:, col["c_hp_lag"]] = best
        ax = np.full(n, -9, np.float32); axabs = np.zeros(n, np.float32)
        for i in range(3):
            for j in range(3):
                c = (da["z_ax"][ia, i] * db["z_ax"][ib, j]).sum(1)
                ax = np.maximum(ax, c); axabs += np.abs(c)
        out[:, col["ax_max"]] = ax; out[:, col["ax_absmean"]] = axabs / 9.0
    else:
        for k in ["c_hp", "c_d", "c_jerk", "c_env", "c_vert", "c_horiz", "c_spec", "c_specj", "c_hp_lag", "ax_max", "ax_absmean"]:
            out[:, col[k]] = M[k][ia, ib]
    out[:, col["d_logstd"]] = da["logstd"][ia] - db["logstd"][ib]
    out[:, col["ad_logstd"]] = np.abs(out[:, col["d_logstd"]])
    out[:, col["d_fdom"]] = np.abs(da["fdom"][ia] - db["fdom"][ib])
    out[:, col["cos_dph"]] = np.cos(da["phase"][ia] - db["phase"][ib]) * (da["fdom"][ia] == db["fdom"][ib])
    out[:, col["std_a"]] = da["std"][ia]; out[:, col["std_b"]] = db["std"][ib]
    out[:, col["fdom_a"]] = da["fdom"][ia]; out[:, col["fdom_b"]] = db["fdom"][ib]
    out[:, col["jm_a"]] = da["jerk_mean"][ia]; out[:, col["jm_b"]] = db["jerk_mean"][ib]
    out[:, col["ga_x"]:col["ga_x"] + 3] = da["gmean"][ia]
    out[:, col["gb_x"]:col["gb_x"] + 3] = db["gmean"][ib]
    out[:, col["gs_a"]] = da["gstd"][ia].mean(1); out[:, col["gs_b"]] = db["gstd"][ib].mean(1)
    out[:, col["pair"]] = pair_id
    return out


def score_matrix(model, da, db, pair_id, M=None, chunk=1_500_000, raw=True):
    """Full N_a x N_b matrix of model log-odds. Computes pair features in chunks."""
    if M is None:
        M = pair_matrices(da, db)
    Na, Nb = M["c_hp"].shape
    S = np.empty((Na, Nb), dtype=np.float32)
    rows_per = max(1, chunk // Nb)
    for r0 in range(0, Na, rows_per):
        r1 = min(Na, r0 + rows_per)
        ia = np.repeat(np.arange(r0, r1), Nb)
        ib = np.tile(np.arange(Nb), r1 - r0)
        X = pair_feature_rows(da, db, ia, ib, pair_id, M=M)
        p = model.predict(X, raw_score=raw, num_threads=6)
        S[r0:r1] = p.reshape(r1 - r0, Nb)
    return S


def simple_score(M):
    """Hand-weighted physics score (no learning) for a baseline."""
    return (M["c_hp_lag"] + M["c_d"] + M["c_jerk"] + M["c_env"] + M["c_vert"] + 0.5 * M["c_spec"] + 0.5 * M["ax_max"]).astype(np.float32)


def hungarian(S, forbid=None):
    """Maximise total score. Returns col index for each row."""
    C = -S.astype(np.float64)
    if forbid is not None:
        C[forbid] = 1e6
    r, c = linear_sum_assignment(C)
    out = np.empty(S.shape[0], dtype=np.int64)
    out[r] = c
    return out


def eval_assignment(assign, true_b_of_a, lab_a, lab_b, dyn_a):
    """assign[i] = chosen b for a-row i. true_b_of_a[i] = correct b. Returns dict of accuracies."""
    exact = assign == true_b_of_a
    same = lab_a == lab_b[assign]
    res = {}
    for name, m in [("all", np.ones_like(dyn_a, dtype=bool)), ("dyn", dyn_a), ("sta", ~dyn_a)]:
        if m.sum() == 0:
            res[name] = (np.nan, np.nan, 0)
        else:
            res[name] = (exact[m].mean(), same[m].mean(), int(m.sum()))
    return res
