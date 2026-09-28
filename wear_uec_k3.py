"""wear-uec-k3 : Kaggle GPU script (no internet, no secrets). 5-fold OOF variant of wear-uec-k1.

Full-train refit (all 22 train subjects, *_2 sessions excluded as UEC does) of two members of the public
UEC-dx2 ensemble, ported from approach_base/src/train_exp024_cnn8_videomae_aux.py:
  M2 "Inertial-CNN8 + Video-MLP"          -> /kaggle/working/test_cnn8.npy     (12234,19) float32
  M1 "Inertial-XceptionTime + Video-MLP"  -> /kaggle/working/test_xcepvid.npy  (12234,19) float32
Soft softmax probabilities in test-id order (row i = test id i).

Manifest hyper-parameters (both members): window 50 / stride 25 / purity 0.8, 4 sensors as separate rows with a
sensor embedding (dim 8), centre-15 raw 768-d VideoMAE frames -> vi_mean / vi_std / vi_delta5 + 56 scalar features,
cnn-base 64, cnn-dropout 0.25, video-hidden 128, video-out 128, video-dropout 0.55, classifier 256 / dropout 0.50,
batch 1024, AdamW lr 5e-4 wd 2e-3, CE with sqrt class weights + label smoothing 0.07.
M1 additionally: --inertial-backbone xceptiontime --xception-nf 16 --xception-kernel-size 40 --xception-adaptive-size 8.

Deviations from the repo (see NOTES.md): fixed epochs (6 for M2, 8 for M1) instead of early stopping on a
validation split, LR halved before epochs 4 and 6 instead of ReduceLROnPlateau, test probabilities re-saved after
every epoch, and a hard wall-time guard (no epoch starts after 45 min).

Local smoke test (CPU):  WEAR_LOCAL=1 WEAR_SMOKE=1  -> a few sessions, 1 epoch, 30 batches per model, output to
E:\\Claude code\\wear\\kaggle\\uec_k1\\smoke_out\\ .
"""
import os, sys, glob, time, json, math

LOCAL = os.environ.get("WEAR_LOCAL", "0") == "1"
SMOKE = os.environ.get("WEAR_SMOKE", "0") == "1"
if LOCAL:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "3")
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
if LOCAL:
    torch.set_num_threads(3)

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

ROOT = r"E:\Claude code\wear\data" if LOCAL else "/kaggle/input/competitions/3rd-wear-dataset-challenge-hasca-2026"
OUT = r"E:\Claude code\wear\kaggle\uec_k3\smoke_out" if LOCAL else "/kaggle/working"
PREP = r"E:\Claude code\wear\data\prep"          # LOCAL only: PCA-160 stand-in for sessions without raw video
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- constants (UEC exp024 defaults / manifest)
WS, STRIDE, PURITY = 50, 25, 0.8
IMU_HZ, VIDEO_HZ, VWIN = 50, 30, 15
NC, VIDEO_DIM, VIDEO_SCALAR_DIM = 19, 768, 56
SEED, BATCH = 42, 1024
GUARD_START_S = int(os.environ.get('WEAR_GUARD_MIN', '54')) * 60                 # never START an epoch after this wall time
LR_HALVE_BEFORE = (4, 6)                # lr *= 0.5 before these epochs (replaces ReduceLROnPlateau)
EPOCHS = {"cnn8": int(os.environ.get("WEAR_EPOCHS_CNN8", "6")), "xcep": int(os.environ.get("WEAR_EPOCHS_XCEP", "8"))}
MAX_BATCHES = None
if SMOKE:
    EPOCHS = {k: 1 for k in EPOCHS}
    MAX_BATCHES = 30
MODELS = [m for m in os.environ.get("WEAR_MODELS", "cnn8,xcep").split(",") if m]
SMOKE_SESSIONS = [s for s in os.environ.get("WEAR_SESSIONS", "sbj_5,sbj_9,sbj_20").split(",") if s]

COMMON = dict(cnn_base_channels=64, cnn_dropout=0.25, video_hidden_dim=128, video_out_dim=128, video_dropout=0.55,
              classifier_hidden=256, classifier_dropout=0.50, sensor_emb_dim=8,
              learning_rate=5e-4, weight_decay=2e-3, label_smoothing=0.07, class_weight="sqrt")
CFG = {
    "cnn8": dict(COMMON, inertial_backbone="cnn8", out="test_cnn8.npy"),
    "xcep": dict(COMMON, inertial_backbone="xceptiontime", xception_nf=16, xception_kernel_size=40,
                 xception_adaptive_size=8, out="test_xcepvid.npy"),
}

LABEL_TO_ID = {"null": 0, "jogging": 1, "jogging (rotating arms)": 2, "jogging (skipping)": 3, "jogging (sidesteps)": 4,
    "jogging (butt-kicks)": 5, "stretching (triceps)": 6, "stretching (lunging)": 7, "stretching (shoulders)": 8,
    "stretching (hamstrings)": 9, "stretching (lumbar rotation)": 10, "push-ups": 11, "push-ups (complex)": 12,
    "sit-ups": 13, "sit-ups (complex)": 14, "burpees": 15, "lunges": 16, "lunges (complex)": 17, "bench-dips": 18}
# UEC SENSOR_COLS order (= default --sensor-keys ra rl ll la) and SENSOR_TO_ID (embedding index)
SENSOR_COLS = {"ra": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"],
               "rl": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
               "ll": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
               "la": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"]}
SENSOR_KEYS = list(SENSOR_COLS)
SENSOR_TO_ID = {"ra": 0, "la": 1, "rl": 2, "ll": 3}
LOC_TO_KEY = {"right_arm": "ra", "left_arm": "la", "right_leg": "rl", "left_leg": "ll",
              "right arm": "ra", "left arm": "la", "right leg": "rl", "left leg": "ll",
              "ra": "ra", "la": "la", "rl": "rl", "ll": "ll"}


# ---------------------------------------------------------------- feature functions (vectorised UEC formulas)
def encode_label(value):            # verbatim approach_base/src/train_inertial_gbdt.encode_label
    if value != value:
        return None
    if isinstance(value, str):
        text = value.strip(); lowered = text.lower()
        if lowered in {"", "null"}:
            return LABEL_TO_ID["null"]
        if lowered in {"nan", "none"}:
            return None
        if text in LABEL_TO_ID:
            return LABEL_TO_ID[text]
        return int(float(text))
    return int(value)


def safe(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def make_cnn8(W, eps=1e-6):         # W (M,50,3) -> (M,8,50)   == make_cnn8_inertial per window
    W = safe(W)
    x, y, z = W[..., 0], W[..., 1], W[..., 2]
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2 + eps)
    d = np.diff(W, axis=1, prepend=W[:, :1]); ad = np.abs(d); dm = np.linalg.norm(d, axis=2)
    return safe(np.stack([x, y, z, mag, ad[..., 0], ad[..., 1], ad[..., 2], dm], 1))


def row_cos(a, b, eps=1e-6):        # == _row_cos
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps)


def stats_1d(x):                    # x (B,L) -> (B,16)   == _stats_1d
    q10, q25, q50, q75, q90 = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    return np.stack([x.mean(1), x.std(1), x.min(1), x.max(1), x.max(1) - x.min(1), q10, q25, q50, q75, q90, q75 - q25,
                     np.sqrt(np.mean(x ** 2, 1)), np.sum(x ** 2, 1), x[:, 0], x[:, -1], x[:, -1] - x[:, 0]], 1)


def video_feats(fr):                # fr (B,15,768) -> vi_mean, vi_std, vi_delta5 (768 each), scalar (56)
    fr = safe(fr)
    vi_mean = fr.mean(1); vi_std = fr.std(1); vi_d5 = fr[:, -5:].mean(1) - fr[:, :5].mean(1)
    first, last = fr[:, 0], fr[:, -1]
    first5, mid5, last5 = fr[:, :5].mean(1), fr[:, 5:10].mean(1), fr[:, 10:].mean(1)
    frame_norm = np.linalg.norm(fr, axis=2)
    diff_norm = np.linalg.norm(np.diff(fr, axis=1), axis=2)
    frame_cos = row_cos(fr[:, :-1], fr[:, 1:])
    sc = np.stack([np.linalg.norm(last - first, axis=1), row_cos(first, last),
                   np.linalg.norm(mid5 - first5, axis=1), np.linalg.norm(last5 - mid5, axis=1), np.linalg.norm(last5 - first5, axis=1),
                   row_cos(first5, mid5), row_cos(mid5, last5), row_cos(first5, last5)], 1)
    scalar = np.concatenate([sc, stats_1d(frame_norm), stats_1d(diff_norm), stats_1d(frame_cos)], 1)
    assert scalar.shape[1] == VIDEO_SCALAR_DIM
    return safe(vi_mean), safe(vi_std), safe(vi_d5), safe(scalar)


# ---------------------------------------------------------------- train data
def read_session(path):
    df = pd.read_csv(path, low_memory=False, keep_default_na=False)
    vals = df["label"].to_numpy()
    m = {}
    for u in set(vals.tolist()):
        e = encode_label(u); m[u] = -1 if e is None else int(e)
    lab = np.fromiter((m[v] for v in vals), dtype=np.int64, count=len(vals))
    A = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
                  for k in SENSOR_KEYS for c in SENSOR_COLS[k]], 1)          # (T,12): ra, rl, ll, la x (x,y,z)
    sid = int(df["sbj_id"].to_numpy()[0]) if "sbj_id" in df.columns else -1
    return lab, A, sid


_PCA = {}
def load_frames(sess, n_samples):
    """(frames,768) VideoMAE array of a session (memmap on Kaggle).  LOCAL fallback: PCA-160 stand-in."""
    path = os.path.join(ROOT, "train", "videomae_feat", f"{sess}.npy")
    need = int(round(n_samples * VIDEO_HZ / IMU_HZ))
    try:
        V = np.load(path, mmap_mode="r")
        if V.ndim == 2 and V.shape[1] == VIDEO_DIM and V.shape[0] >= need - 2 * VIDEO_HZ:
            return V, "raw"
        msg = f"unexpected shape {V.shape} (need ~{need})"
    except Exception as e:
        msg = repr(e)
    if not LOCAL:
        raise RuntimeError(f"video for {sess}: {msg}")
    log(f"  {sess}: raw video unavailable ({msg}); PCA-160 stand-in from {PREP}")
    if not _PCA:
        _PCA["meta"] = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
        _PCA["z"] = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
        _PCA["mean"] = np.load(os.path.join(PREP, "pca_mean.npy")).astype(np.float32)
        _PCA["comps"] = np.load(os.path.join(PREP, "pca_components.npy")).astype(np.float32)
    meta = _PCA["meta"]; rows = np.where(meta["session"].to_numpy() == sess)[0]
    ts = meta["t"].to_numpy()[rows].astype(np.int64)
    fr = (_PCA["mean"] + np.asarray(_PCA["z"][rows], np.float32) @ _PCA["comps"]).astype(np.float32)   # (n,15,768)
    V = np.full((need, VIDEO_DIM), np.nan, np.float32)      # only the centre-15 frames of each tile exist
    idx = ts[:, None] * VIDEO_HZ + np.arange(8, 8 + VWIN)[None, :]
    ok = idx[:, -1] < need
    V[idx[ok].ravel()] = fr[ok].reshape(-1, VIDEO_DIM)
    return V, "pca"


def build_session(sess):
    lab, A, sid = read_session(os.path.join(ROOT, "train", "inertial_feat", f"{sess}.csv"))
    T = len(lab)
    starts = np.arange(0, T - WS + 1, STRIDE)
    # label rule == assign_window_label_from_array(purity 0.8): any None -> skip; majority ratio >= 0.8
    oh = np.zeros((T + 1, NC), np.int32); v = lab >= 0
    oh[np.arange(T)[v] + 1, lab[v]] = 1
    cs = np.cumsum(oh, 0); cnt = cs[starts + WS] - cs[starts]
    ok_lab = (cnt.sum(1) == WS) & (cnt.max(1) / float(WS) >= PURITY)
    y = cnt.argmax(1)
    # sensor rule == has_only_finite_sensor_windows over all 4 sensors
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(A).all(1))])
    ok_sen = (bad[starts + WS] - bad[starts]) == 0
    # video: centre crop of 15 frames around the window centre (frame = round(sample*30/50)); non-finite -> skip
    V, src = load_frames(sess, T)
    centre = np.rint((starts + WS / 2.0) * VIDEO_HZ / IMU_HZ).astype(np.int64)
    lo = centre - VWIN // 2
    ok_rng = (lo >= 0) & (lo + VWIN <= V.shape[0])
    keep = ok_lab & ok_sen & ok_rng
    ks, lo_k, y_k = starts[keep], lo[keep], y[keep]
    fidx = (lo_k[:, None] + np.arange(VWIN)[None, :]).ravel()
    frames = np.asarray(V[fidx], np.float32).reshape(len(ks), VWIN, VIDEO_DIM)
    fin = np.isfinite(frames).all(axis=(1, 2))
    ks, y_k, frames = ks[fin], y_k[fin], frames[fin]
    vm, vs, vd, sc = video_feats(frames); del frames
    W = A[(ks[:, None] + np.arange(WS)[None, :]).ravel()].reshape(len(ks), WS, 4, 3).transpose(0, 2, 1, 3)  # (n,4,50,3)
    X = np.stack([make_cnn8(W[:, k]) for k in range(4)], 1)                                                  # (n,4,8,50)
    st = dict(session=sess, sbj=sid, samples=int(T), windows=int(len(starts)), kept=int(len(ks)),
              skipped_label=int((~ok_lab).sum()), skipped_sensor=int((ok_lab & ~ok_sen).sum()),
              skipped_video=int((ok_lab & ok_sen).sum() - len(ks)), video=src)
    return X, y_k, vm, vs, vd, sc, st


# ---------------------------------------------------------------- test data (row i == test id i)
def build_test():
    tm = pd.read_csv(os.path.join(ROOT, "test", "test_meta_data.csv"))
    ids = tm["id"].to_numpy()
    assert (ids == np.arange(len(tm))).all(), "test_meta ids are not 0..N-1 in order"
    TI = np.load(os.path.join(ROOT, "test", "test_inertial_data.npy")).astype(np.float32)
    if TI.shape[1] == 3 and TI.shape[2] != 3:
        TI = TI.transpose(0, 2, 1)                       # fix_test_inertial_window
    assert TI.shape[1:] == (WS, 3), TI.shape
    Xte = make_cnn8(TI)
    sens = np.array([SENSOR_TO_ID[LOC_TO_KEY[str(s).lower().strip()]] for s in tm["sensor_location"].to_numpy()], np.int64)
    TV = np.load(os.path.join(ROOT, "test", "test_videomae_data.npy"), mmap_mode="r")      # (N,768,15)
    feats = [[], [], [], []]
    for b in range(0, len(tm), 2048):
        fr = np.asarray(TV[b:b + 2048], np.float32)
        if fr.shape[1] == VIDEO_DIM and fr.shape[2] != VIDEO_DIM:
            fr = fr.transpose(0, 2, 1)                   # -> (n,15,768)
        for lst, f in zip(feats, video_feats(fr)):
            lst.append(f)
    vm, vs, vd, sc = [np.concatenate(l) for l in feats]
    return Xte, sens, vm, vs, vd, sc, len(tm)


# ---------------------------------------------------------------- model (structure verbatim from the repo)
class ConvBlock(nn.Module):
    def __init__(self, i, o, k, dropout):
        super().__init__(); p = k // 2
        self.net = nn.Sequential(nn.Conv1d(i, o, k, padding=p, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Conv1d(o, o, k, padding=p, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)


# --- tsai XceptionTime re-implementation (repo's models/xceptiontime.py is not public)
class ConvBN(nn.Sequential):        # tsai ConvBlock(ni, nf, 1[, act=None]): conv(bias=False) -> BN -> [ReLU]
    def __init__(self, ni, nf, act=True):
        layers = [nn.Conv1d(ni, nf, 1, bias=False), nn.BatchNorm1d(nf)]
        if act:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class SeparableConv1d(nn.Module):
    def __init__(self, ni, nf, ks):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(ni, ni, ks, padding=ks // 2, groups=ni, bias=False)
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, bias=False)
    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class XceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()
        ks = [ks // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else nn.Identity()
        self.convs = nn.ModuleList([SeparableConv1d(nf if bottleneck else ni, nf, k) for k in ks])
        self.maxconvpool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False))
    def forward(self, x):
        b = self.bottleneck(x)
        return torch.cat([l(b) for l in self.convs] + [self.maxconvpool(x)], 1)


class XceptionBlock(nn.Module):
    def __init__(self, ni, nf, residual=True, **kw):
        super().__init__()
        self.residual = residual
        self.xception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        n_in = n_out = None
        for i in range(4):
            if self.residual and (i - 1) % 2 == 0:
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else ConvBN(n_in, n_out * 4 * 2, act=False))
            n_out = nf * 2 ** i
            n_in = ni if i == 0 else n_out * 2
            self.xception.append(XceptionModule(n_in, n_out, **kw))
        self.act = nn.ReLU()
    def forward(self, x):
        res = x
        for i in range(4):
            x = self.xception[i](x)
            if self.residual and (i + 1) % 2 == 0:
                res = x = self.act(x + self.shortcut[i // 2](res))
        return x


class XceptionTime(nn.Module):
    def __init__(self, c_in, c_out, nf=16, adaptive_size=50, residual=True, bottleneck=True, ks=40):
        super().__init__()
        self.block = XceptionBlock(c_in, nf, residual=residual, ks=ks, bottleneck=bottleneck)
        hn = nf * 32
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size), ConvBN(hn, hn // 2), ConvBN(hn // 2, hn // 4),
                                  ConvBN(hn // 4, c_out), nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(self, x):
        return self.head(self.block(x))


def make_video_projection(a):
    vh = a["video_hidden_dim"]; vd = a["video_dropout"]
    return nn.ModuleDict({
        "mean": nn.Sequential(nn.Linear(VIDEO_DIM, vh), nn.LayerNorm(vh), nn.ReLU(), nn.Dropout(vd)),
        "std": nn.Sequential(nn.Linear(VIDEO_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(vd)),
        "delta5": nn.Sequential(nn.Linear(VIDEO_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(vd)),
        "scalar": nn.Sequential(nn.Linear(VIDEO_SCALAR_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(vd)),
        "out": nn.Sequential(nn.Linear(vh + vh // 2 + vh // 2 + vh // 2, a["video_out_dim"]), nn.LayerNorm(a["video_out_dim"]), nn.ReLU(), nn.Dropout(vd)),
    })


class CNN8VideoAuxNet(nn.Module):
    def __init__(self, a):
        super().__init__()
        base = a["cnn_base_channels"]
        self.inertial_feature_dim = base * 2 * 2
        if a["inertial_backbone"] == "cnn8":
            self.cnn = nn.Sequential(ConvBlock(8, base, 5, a["cnn_dropout"]), nn.MaxPool1d(2),
                                     ConvBlock(base, base * 2, 3, a["cnn_dropout"]), nn.MaxPool1d(2),
                                     ConvBlock(base * 2, base * 2, 3, a["cnn_dropout"]))
            self.xception = None
        else:
            self.cnn = None
            self.xception = XceptionTime(c_in=8, c_out=self.inertial_feature_dim, nf=a["xception_nf"],
                                         adaptive_size=a["xception_adaptive_size"], residual=True, bottleneck=True,
                                         ks=a["xception_kernel_size"])
        self.sensor_emb = nn.Embedding(len(SENSOR_TO_ID), a["sensor_emb_dim"])
        total = self.inertial_feature_dim + a["video_out_dim"] + a["sensor_emb_dim"]
        self.video_projection = make_video_projection(a)
        h = a["classifier_hidden"]; cd = a["classifier_dropout"]
        self.classifier = nn.Sequential(nn.Linear(total, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(cd),
                                        nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(cd),
                                        nn.Linear(h // 2, NC))

    def forward(self, x, vm, vs, vd, sc, sens):
        if self.cnn is not None:
            h = self.cnn(x)
            inert = torch.cat([F.adaptive_avg_pool1d(h, 1).squeeze(-1), F.adaptive_max_pool1d(h, 1).squeeze(-1)], 1)
        else:
            inert = self.xception(x)
        p = self.video_projection
        v = p["out"](torch.cat([p["mean"](vm), p["std"](vs), p["delta5"](vd), p["scalar"](sc)], 1))
        return self.classifier(torch.cat([inert, v, self.sensor_emb(sens)], 1))



# ---------------------------------------------------------------- tiles (every 1-s window, all 4 sensors) for OOF
FOLDS = {0: [5, 14, 17, 21], 1: [0, 4, 9, 15], 2: [3, 7, 11, 12, 19], 3: [1, 8, 10, 16], 4: [2, 6, 13, 18, 20]}
if SMOKE:
    FOLDS = {0: [5], 1: [9], 2: [20]}


def build_tiles(sess):
    lab, A, sid = read_session(os.path.join(ROOT, "train", "inertial_feat", f"{sess}.csv"))
    T = len(lab)
    V, src = load_frames(sess, T)
    ts = np.arange(T // WS); starts = ts * WS
    centre = np.rint((starts + WS / 2.0) * VIDEO_HZ / IMU_HZ).astype(np.int64); lo = centre - VWIN // 2
    ok = (lo >= 0) & (lo + VWIN <= V.shape[0])
    ts, starts, lo = ts[ok], starts[ok], lo[ok]
    n = len(ts)
    fidx = (lo[:, None] + np.arange(VWIN)[None, :]).ravel()
    frames = np.asarray(V[fidx], np.float32).reshape(n, VWIN, VIDEO_DIM)
    vvalid = np.isfinite(frames).all(axis=(1, 2))
    vm, vs, vd, sc = video_feats(frames); del frames
    W = A[(starts[:, None] + np.arange(WS)[None, :]).ravel()].reshape(n, WS, 4, 3).transpose(0, 2, 1, 3)   # (n,4,50,3)
    valid = np.isfinite(W).all(axis=(2, 3))                                                                  # (n,4)
    X = np.stack([make_cnn8(W[:, k]) for k in range(4)], 1)                                                  # (n,4,8,50)
    return dict(session=sess, sbj=sid, t=ts, X=X, vm=vm, vs=vs, vd=vd, sc=sc, valid=valid, vvalid=vvalid)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    log(f"device={dev} LOCAL={LOCAL} SMOKE={SMOKE} epochs={EPOCHS} max_batches={MAX_BATCHES} models={MODELS} folds={FOLDS}")
    all_sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "train", "inertial_feat", "sbj_*.csv")))
    train_sessions = [s for s in all_sessions if not (len(s.split("_")) >= 3 and s.split("_")[-1] == "2")]
    if SMOKE:
        all_sessions = [s for s in SMOKE_SESSIONS if s in all_sessions]; train_sessions = [s for s in train_sessions if s in all_sessions]
    log(f"sessions all={len(all_sessions)} train={len(train_sessions)}")

    parts, stats, psbj = [], [], []
    for s in train_sessions:
        X, y, vm, vs, vd, sc, st = build_session(s)
        parts.append((X, y, vm, vs, vd, sc)); stats.append(st); psbj.append(np.full(len(y), st["sbj"], np.int64)); log(st)
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    VM = np.concatenate([p[2] for p in parts]); VS = np.concatenate([p[3] for p in parts])
    VD = np.concatenate([p[4] for p in parts]); SC = np.concatenate([p[5] for p in parts]); WSBJ = np.concatenate(psbj)
    del parts
    Nw = len(y)
    Xs = np.ascontiguousarray(X.transpose(1, 0, 2, 3)).reshape(4 * Nw, 8, WS); del X
    tile = np.tile(np.arange(Nw), 4)
    sens = np.repeat(np.array([SENSOR_TO_ID[k] for k in SENSOR_KEYS], np.int64), Nw)
    ys = np.tile(y, 4); ssbj = np.tile(WSBJ, 4)
    log(f"windows={Nw} samples={len(ys)} subjects={sorted(set(WSBJ.tolist()))}")

    tiles = [build_tiles(s) for s in all_sessions]
    rows = pd.DataFrame({"session": np.concatenate([[d["session"]] * len(d["t"]) for d in tiles]),
                         "sbj": np.concatenate([[d["sbj"]] * len(d["t"]) for d in tiles]),
                         "t": np.concatenate([d["t"] for d in tiles])})
    rows.to_csv(os.path.join(OUT, "oof_rows.csv"), index=False)
    NR = len(rows); off = np.r_[0, np.cumsum([len(d["t"]) for d in tiles])]
    log(f"tile rows={NR} (expected 69326 on full data)")
    tile_valid = np.concatenate([d["valid"] for d in tiles]); np.save(os.path.join(OUT, "oof_valid.npy"), tile_valid)

    Xte, sens_te, tVM, tVS, tVD, tSC, NT = build_test()
    log(f"test rows={NT}")
    OOF = {m: np.full((NR, 4, NC), np.nan, np.float32) for m in MODELS}
    TEST = {m: [] for m in MODELS}
    summary = dict(sessions=stats, windows=int(Nw), tile_rows=int(NR), epochs=EPOCHS, folds={k: v for k, v in FOLDS.items()}, runs=[])

    def save_all():
        for m in MODELS:
            np.save(os.path.join(OUT, f"oof_{m}.npy"), OOF[m])
            if TEST[m]:
                np.save(os.path.join(OUT, f"test_{m}_folds.npy"), np.stack(TEST[m]))
                np.save(os.path.join(OUT, CFG[m]["out"]), np.mean(TEST[m], 0).astype(np.float32))
        json.dump(summary, open(os.path.join(OUT, "k3_summary.json"), "w"), indent=1)

    @torch.inference_mode()
    def predict(model, feats, norm):
        model.eval(); outs = []
        cm, cs, sm, ss = norm
        Xf, vm, vs, vd, sc, sn = feats
        for b in range(0, len(sn), 4096):
            s = slice(b, b + 4096)
            lo = model(torch.from_numpy((Xf[s] - cm) / cs).to(dev), torch.from_numpy(vm[s]).to(dev), torch.from_numpy(vs[s]).to(dev),
                       torch.from_numpy(vd[s]).to(dev), torch.from_numpy((sc[s] - sm) / ss).to(dev), torch.from_numpy(sn[s]).to(dev))
            outs.append(torch.softmax(lo.float(), 1).cpu())
        return torch.cat(outs).numpy().astype(np.float32)

    for f, held in FOLDS.items():
        if time.time() - T0 > GUARD_START_S:
            log(f"fold {f}: time guard, skipped"); break
        tr = ~np.isin(ssbj, held); n = int(tr.sum())
        cnn_mean = Xs[tr].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
        cnn_std = np.maximum(Xs[tr].std(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None], 1e-6)
        tw = tile[tr]; sc_mean = SC[tw].mean(0, dtype=np.float64).astype(np.float32); sc_std = np.maximum(SC[tw].std(0, dtype=np.float64).astype(np.float32), 1e-6)
        norm = (cnn_mean, cnn_std, sc_mean, sc_std)
        def to_dev(a, dt):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dt).to(dev)
        train_t = dict(X=to_dev((Xs[tr] - cnn_mean) / cnn_std, torch.float32), VM=to_dev(VM, torch.float32), VS=to_dev(VS, torch.float32),
                       VD=to_dev(VD, torch.float32), SC=to_dev((SC - sc_mean) / sc_std, torch.float32),
                       tile=to_dev(tile[tr], torch.long), sens=to_dev(sens[tr], torch.long), y=to_dev(ys[tr], torch.long))
        held_sessions = [i for i, d in enumerate(tiles) if d["sbj"] in held]
        log(f"== fold {f} held={held} train samples={n} held sessions={[tiles[i]['session'] for i in held_sessions]}")
        for name in MODELS:
            if time.time() - T0 > GUARD_START_S:
                log(f"fold {f} {name}: time guard, skipped"); break
            cfg = CFG[name]
            torch.manual_seed(SEED + f); np.random.seed(SEED + f)
            model = CNN8VideoAuxNet(cfg).to(dev)
            counts = np.maximum(np.bincount(ys[tr], minlength=NC).astype(np.float32), 1.0)
            w = np.sqrt(n / (NC * counts)); w = w / w.mean()
            crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=cfg["label_smoothing"])
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            g = torch.Generator().manual_seed(SEED + f); lr = cfg["learning_rate"]; hist = []
            for ep in range(1, EPOCHS[name] + 1):
                if time.time() - T0 > GUARD_START_S:
                    log(f"fold {f} {name}: time guard, not starting epoch {ep}"); break
                if ep in LR_HALVE_BEFORE:
                    lr *= 0.5
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                model.train(); te = time.time(); tl = 0.0; nb = 0
                order = torch.randperm(n, generator=g).to(dev)
                for b in range(0, n, BATCH):
                    if MAX_BATCHES is not None and nb >= MAX_BATCHES:
                        break
                    idx = order[b:b + BATCH]
                    if len(idx) < 2:
                        continue
                    tt = train_t["tile"][idx]
                    logits = model(train_t["X"][idx], train_t["VM"][tt], train_t["VS"][tt], train_t["VD"][tt], train_t["SC"][tt], train_t["sens"][idx])
                    loss = crit(logits, train_t["y"][idx])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    tl += float(loss.item()); nb += 1
                hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), lr=lr, batches=nb, epoch_sec=round(time.time() - te, 1)))
                log(f"fold {f} {name} epoch={ep} loss={tl / max(nb, 1):.4f} epoch_sec={time.time() - te:.0f} elapsed={time.time() - T0:.0f}s")
            # OOF: every tile of the held-out subjects' sessions, all 4 sensors
            for i in held_sessions:
                d = tiles[i]; nt = len(d["t"])
                for k, key in enumerate(SENSOR_KEYS):
                    P = predict(model, (d["X"][:, k], d["vm"], d["vs"], d["vd"], d["sc"], np.full(nt, SENSOR_TO_ID[key], np.int64)), norm)
                    P[~d["valid"][:, k]] = np.nan
                    OOF[name][off[i]:off[i + 1], k] = P
            Pt = predict(model, (Xte, tVM, tVS, tVD, tSC, sens_te), norm); TEST[name].append(Pt)
            acc = [np.nanmean(np.nanmean(OOF[name][off[i]:off[i + 1]], 1).argmax(1) == 0) for i in held_sessions[:1]]
            summary["runs"].append(dict(fold=f, model=name, epochs=len(hist), history=hist, elapsed=round(time.time() - T0)))
            log(f"fold {f} {name}: OOF done, test pred_dist={np.bincount(Pt.argmax(1), minlength=NC).tolist()}")
            del model, opt
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del train_t
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        save_all()
    save_all()
    log(f"DONE elapsed {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
