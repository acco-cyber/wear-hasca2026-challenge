"""Port of WEAR2026-UEC-dx2 approach_base/src/train_video_mlp.py + train_video_cnn.py (+ the missing
approach_base/src/video_only_common.py, reconstructed) for our CPU box.

Recipe kept from the repo (train_video_mlp.py / train_video_cnn.py defaults + manifest command):
  windows 50 samples / stride 25 (stride 50 on the PCA backend), label = majority with purity >= 0.8,
  window kept only if the ra sensor is finite (default --sensor-keys ra -> one sample per window),
  video crop = 15 frames centred on the 1-s window (30 fps), --exclude-file-id-suffix-2,
  frame selection first_mid_last (frames 0,7,14) + delta_concat for member A / CNN,
  MLP: Linear(in,1024)-ReLU-Dropout(0.3)-Linear(1024,19); AdamW lr 1e-3 wd 1e-4, batch 256,
  CE with balanced class weights, ReduceLROnPlateau(max, 0.5, patience 3) on val macro-F1,
  early stopping 5 on val macro-F1, <=100 epochs, seed 42.
  CNN: Conv1d(in,256,1)-BN-ReLU-Conv(256,256,3)-BN-ReLU-Conv(256,256,3)-BN-ReLU-GAP-Linear(256,256)-ReLU-Drop(0.2)-Linear.
Reconstructed (video_only_common.py is not in the repo): build_video_mlp_feature_vector =
  [mean_t, std_t, mean(last k) - mean(first k), 56 scalar stats], k = max(1, T//3) -- identical to the
  repo's exp024 make_video_aggregate_features/make_video_scalar_features at T=15; scalar block z-scored
  with train statistics (as exp024 does); delta_concat = concat([x, diff(x) with zero first row], -1).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys, json, time, glob, argparse
import numpy as np, pandas as pd

DATA = r"E:\Claude code\wear\data"
PREP = os.path.join(DATA, "prep")
OUT = r"E:\Claude code\wear\uec\video"
FEAT = os.path.join(OUT, "feats")
NC = 19
CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)", "jogging (butt-kicks)",
               "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
               "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups", "sit-ups (complex)", "burpees",
               "lunges", "lunges (complex)", "bench-dips"]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
IMU_HZ, VIDEO_HZ, WIN, VWIN = 50, 30, 50, 15
SEL_FML = [0, VWIN // 2, VWIN - 1]          # first_mid_last -> frames 0, 7, 14


# ----------------------------------------------------------------------------- video_only_common (reconstructed)
def center_crop_indices(start, window_size=WIN, video_hz=VIDEO_HZ, video_window_size=VWIN):
    center = int(round((start + window_size / 2.0) * video_hz / IMU_HZ))
    s0 = center - video_window_size // 2
    return s0, s0 + video_window_size       # start=0 -> [8, 23) (same as our kaggle prep)


def normalize_video_window(v):
    """(768,15) test layout or (15,768) train layout -> (T,768) float32, NaN->0."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 2 and v.shape[0] == 768 and v.shape[1] != 768:
        v = v.T
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def select_frames(v, selection):
    """v (B,T,D)."""
    if selection == "all":
        return v
    if selection == "first_mid_last":
        T = v.shape[1]
        return v[:, [0, T // 2, T - 1]]
    raise ValueError(selection)


def build_video_temporal_features(v, mode):
    """v (B,T,D) -> (B,T,D) raw or (B,T,2D) delta_concat."""
    if mode == "raw":
        return v
    if mode == "delta_concat":
        d = np.zeros_like(v)
        d[:, 1:] = v[:, 1:] - v[:, :-1]
        return np.concatenate([v, d], axis=-1)
    raise ValueError(mode)


def _rowcos(a, b, eps=1e-6):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps)


def _stats1d(x):
    """x (B,n) -> (B,16), same list as exp024 _stats_1d."""
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    mn, mx = x.min(1), x.max(1)
    return np.stack([x.mean(1), x.std(1), mn, mx, mx - mn, q[0], q[1], q[2], q[3], q[4], q[3] - q[1],
                     np.sqrt((x ** 2).mean(1)), (x ** 2).sum(1), x[:, 0], x[:, -1], x[:, -1] - x[:, 0]], 1)


def build_video_mlp_feature_vector(v):
    """v (B,T,D) -> agg (B,3D) [mean, std, delta_k] and scalar (B,56)."""
    v = v.astype(np.float32, copy=False)
    B, T, D = v.shape
    k = max(1, T // 3)
    first, last = v[:, 0], v[:, -1]
    f5 = v[:, :k].mean(1); m5 = v[:, k:T - k].mean(1) if T - 2 * k > 0 else v[:, k]; l5 = v[:, T - k:].mean(1)
    agg = np.concatenate([v.mean(1), v.std(1), l5 - f5], axis=1)
    frame_norm = np.linalg.norm(v, axis=-1)
    diff_norm = np.linalg.norm(np.diff(v, axis=1), axis=-1)
    frame_cos = _rowcos(v[:, :-1], v[:, 1:])
    sc = np.stack([np.linalg.norm(last - first, axis=-1), _rowcos(first, last),
                   np.linalg.norm(m5 - f5, axis=-1), np.linalg.norm(l5 - m5, axis=-1), np.linalg.norm(l5 - f5, axis=-1),
                   _rowcos(f5, m5), _rowcos(m5, l5), _rowcos(f5, l5)], 1)
    scalar = np.concatenate([sc, _stats1d(frame_norm), _stats1d(diff_norm), _stats1d(frame_cos)], 1)
    return np.nan_to_num(agg).astype(np.float32), np.nan_to_num(scalar).astype(np.float32)


def member_features(win15, member):
    """win15 (B,15,768) -> dict of arrays for one member."""
    if member == "a":      # first_mid_last + delta_concat MLP
        t = build_video_temporal_features(select_frames(win15, "first_mid_last"), "delta_concat")
        agg, sc = build_video_mlp_feature_vector(t)
        return agg, sc
    if member == "b":      # all frames, raw MLP (repo defaults)
        t = build_video_temporal_features(select_frames(win15, "all"), "raw")
        return build_video_mlp_feature_vector(t)
    raise ValueError(member)


# ----------------------------------------------------------------------------- labels / windows
def encode_labels(series):
    s = series.astype(str).str.strip()
    low = s.str.lower()
    out = np.full(len(s), -1, np.int64)
    m = s.map(LABEL_TO_ID)
    ok = m.notna().to_numpy()
    out[ok] = m[ok].astype(np.int64).to_numpy()
    out[(low.isin(["", "null"])).to_numpy()] = 0
    return out


def window_labels(lab, starts, window=WIN, purity=0.8):
    """majority label + purity for each start; -1 if any unlabeled sample."""
    idx = starts[:, None] + np.arange(window)[None, :]
    L = lab[idx]
    bad = (L < 0).any(1)
    Lc = np.where(L < 0, 0, L)
    cnt = np.zeros((len(starts), NC), np.int32)
    for c in range(NC):
        cnt[:, c] = (Lc == c).sum(1)
    y = cnt.argmax(1)
    pur = cnt.max(1) / float(window)
    y = np.where(bad, -1, y)
    return y.astype(np.int64), pur.astype(np.float32)


# ----------------------------------------------------------------------------- torch models / training
def get_torch():
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    return torch


def make_model(kind, in_dim, torch):
    nn = torch.nn
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(1024, NC))

    class VideoCNN(nn.Module):
        def __init__(self, input_feature_dim, proj_dim=256, channels=256, hidden_dim=256, dropout=0.2):
            super().__init__()
            self.input_proj = nn.Conv1d(input_feature_dim, proj_dim, kernel_size=1)
            self.encoder = nn.Sequential(nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
                                         nn.Conv1d(proj_dim, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
                                         nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
                                         nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Linear(channels, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, NC))

        def forward(self, video):
            x = self.input_proj(video.transpose(1, 2))
            return self.head(self.encoder(x).squeeze(-1))
    return VideoCNN(in_dim)


def balanced_class_weights(y):
    counts = np.bincount(y, minlength=NC).astype(np.float64)
    nz = counts > 0
    w = np.ones(NC, np.float64)
    w[nz] = len(y) / (float(nz.sum()) * counts[nz])
    return w.astype(np.float32)


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def seed_everything(seed, torch):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def predict(model, X, torch, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[s:s + bs], np.float32))
            out.append(torch.softmax(model(xb), 1).numpy())
    return np.concatenate(out).astype(np.float32)


def train_member(kind, Xtr, ytr, Xva=None, yva=None, fixed_epochs=None, seed=42, log=print, max_epochs=100,
                 lr=1e-3, wd=1e-4, bs=256, patience=5, sched_patience=3):
    """Mirrors train_video_mlp.main(): early stopping + plateau LR on val macro-F1 (if val given),
    else a fixed number of epochs at constant LR (full-data fit)."""
    torch = get_torch()
    seed_everything(seed, torch)
    in_dim = Xtr.shape[-1]
    model = make_model(kind, in_dim, torch)
    crit = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(balanced_class_weights(ytr)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=sched_patience) if Xva is not None else None
    n = len(ytr)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    g = np.random.RandomState(seed)
    best, best_state, best_ep, bad, hist = -1.0, None, None, 0, []
    n_ep = fixed_epochs if fixed_epochs is not None else max_epochs
    for ep in range(1, n_ep + 1):
        t0 = time.time()
        model.train()
        perm = g.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = np.sort(perm[s:s + bs])
            xb = torch.from_numpy(np.asarray(Xtr[b], np.float32))
            loss = crit(model(xb), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        rec = {"epoch": ep, "train_loss": tot / n, "sec": round(time.time() - t0, 1)}
        if Xva is not None:
            pv = predict(model, Xva, torch)
            f = macro_f1(yva, pv.argmax(1))
            sched.step(f)
            rec.update(val_macro_f1=f, lr=opt.param_groups[0]["lr"])
            if f > best:
                best, best_ep, bad = f, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        log(json.dumps(rec))
        if Xva is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep if best_ep is not None else n_ep, "best_val_macro_f1": best, "history": hist}
