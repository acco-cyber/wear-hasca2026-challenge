"""ctxF shared helpers: compact IMU tile features, macro-F1, log-softmax, chain walks + context aggregation."""
import os, sys
import numpy as np

W = r"E:\Claude code\wear"
D = os.path.join(W, "exp", "v4", "w25run", "ctxF")
CACHE = os.path.join(D, "cache")
FEAS = os.path.join(W, "exp", "v4", "research2", "w25feas")
K7 = os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4")
K9 = os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4")
N_CLS = 19
# 2025 limb index (["left_arm","left_leg","right_arm","right_leg"]) of each pipeline sensor (0 right_arm, 1 right_leg, 2 left_leg, 3 left_arm)
PIPE_TO_25 = np.array([2, 3, 1, 0])
BANDS = [(1, 2), (3, 4), (5, 6), (7, 9), (10, 14), (15, 25)]


def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1); present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def _chan_feats(c):
    """c (n, 50) one channel -> list of (n,) features"""
    n = len(c); out = []
    q = np.quantile(c, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1)
    mu = c.mean(1); sd = c.std(1)
    out += [mu, sd, c.min(1), c.max(1), *q, np.abs(np.diff(c, axis=1)).mean(1), np.abs(np.diff(c, 2, axis=1)).mean(1)]
    z = c - mu[:, None]
    P = np.abs(np.fft.rfft(z, axis=1)) ** 2                    # bins 0..25 Hz (1 Hz each)
    tot = P[:, 1:].sum(1) + 1e-8
    out += [np.log(tot)]
    for lo, hi in BANDS:
        out.append(P[:, lo:hi + 1].sum(1) / tot)
    out.append(1.0 + P[:, 1:].argmax(1).astype(np.float64))
    pn = P[:, 1:] / tot[:, None]; out.append(-(pn * np.log(pn + 1e-12)).sum(1))
    var = (z ** 2).sum(1) + 1e-8
    ac = np.stack([(z[:, lag:] * z[:, :-lag]).sum(1) / var for lag in range(4, 41)], 1)
    out += [ac.max(1), 4.0 + ac.argmax(1), ac[:, 0], ac.min(1)]
    return out


def tile_feats(T, limb):
    """T (n, 50, 3) float -> (n, d) float32. Per axis + magnitude: moments, quantiles, diffs, FFT bands, autocorr; gravity
    direction, axis correlations, limb id."""
    T = np.asarray(T, np.float64); n = len(T)
    mag = np.linalg.norm(T, axis=2)
    cols = []
    for k in range(3):
        cols += _chan_feats(T[:, :, k])
    cols += _chan_feats(mag)
    g = T.mean(1); gn = np.linalg.norm(g, axis=1) + 1e-8
    cols += [g[:, 0] / gn, g[:, 1] / gn, g[:, 2] / gn, gn]
    Z = T - g[:, None]; sd = Z.std(1) + 1e-8
    for a_, b_ in ((0, 1), (0, 2), (1, 2)):
        cols.append((Z[:, :, a_] * Z[:, :, b_]).mean(1) / (sd[:, a_] * sd[:, b_]))
    # first/last-half change (trend within the second)
    cols += [T[:, 25:, k].mean(1) - T[:, :25, k].mean(1) for k in range(3)]
    cols.append(np.full(n, float(limb)))
    return np.stack(cols, 1).astype(np.float32)


def preds_arrays(succ, conf):
    """predecessor and confidence of the link INTO each node"""
    n = len(succ); pr = np.full(n, -1, np.int64); ok = succ >= 0; pr[succ[ok]] = np.flatnonzero(ok)
    cb = np.zeros(n, np.float32); cb[succ[ok]] = conf[ok]
    return pr, cb


def walks(su, cf, start, Wn):
    """from nodes `start` follow su up to Wn steps: nodes (m, Wn) and cumulative confidence (m, Wn)"""
    m = len(start); F = np.full((m, Wn), -1, np.int64); C = np.zeros((m, Wn), np.float32)
    cur = np.asarray(start, np.int64).copy(); c = np.ones(m, np.float32)
    for k in range(Wn):
        ok = cur >= 0; nx = np.full(m, -1, np.int64); nx[ok] = su[cur[ok]]
        c = np.where(nx >= 0, c * np.where(ok, cf[np.maximum(cur, 0)], 0), 0).astype(np.float32)
        F[:, k] = nx; C[:, k] = c; cur = nx
    return F, C


def context(lp_rows, su, cf, start, Wn, decay=0.9, self_w=1.0, mode="logp", conf_pow=1.0):
    """context log-prob for nodes `start`: weighted mean of the row log-probs (or probs) of the node itself and of the nodes
    reached along the chain within Wn steps each way; weight = cumulative confidence^conf_pow * decay^k"""
    pr, cb = preds_arrays(su, cf)
    Ff, Cf = walks(su, cf, start, Wn); Fb, Cb = walks(pr, cb, start, Wn)
    X = lp_rows if mode == "logp" else np.exp(lp_rows)
    acc = self_w * X[start].astype(np.float64); wsum = np.full(len(start), self_w, np.float64)
    for F, C in ((Ff, Cf), (Fb, Cb)):
        for k in range(Wn):
            w = (C[:, k].astype(np.float64) ** conf_pow) * decay ** (k + 1) * (F[:, k] >= 0)
            acc += w[:, None] * X[np.maximum(F[:, k], 0)]; wsum += w
    acc /= wsum[:, None]
    out = lsm(acc) if mode == "logp" else np.log(np.maximum(acc, 1e-9))
    return out.astype(np.float32), wsum
