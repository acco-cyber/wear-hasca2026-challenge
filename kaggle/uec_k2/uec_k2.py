"""wear-uec-k2 -- full-data refit (all 22 train subjects, *_2 sessions excluded) of three members of the public
UEC-dx2 WEAR ensemble; writes SOFT test probabilities (12234,19) float32 in test-id order:
  M6 Inertial-XceptionTime (approach_XceptionTime/src/train_inertial_xceptiontime.py, manifest flags) -> test_xcep.npy
  M5 Video-MLP  (approach_base/src/train_video_mlp.py, first_mid_last + delta_concat)                -> test_vmlp.npy
  M4 Video-CNN  (approach_base/src/train_video_cnn.py, first_mid_last + delta_concat)                -> test_vcnn.npy
Members run in that order (M6 carries the biggest ensemble weight).  Each member's test probabilities are re-saved
after EVERY epoch, so a partial run still leaves usable files.  No epoch is started after WEAR_GUARD_MIN (45) minutes
of wall time, and each member has its own soft deadline so the later members always get some time.
Env: WEAR_LOCAL=1 WEAR_SMOKE=1 runs the whole pipeline on the local prep tiles (3 sessions, 1 epoch, 30 batches).
Deviations from the repo are listed in NOTES.md next to this file.
"""
import os, sys, glob, json, math, time, contextlib

LOCAL = os.environ.get("WEAR_LOCAL", "0") == "1"
SMOKE = os.environ.get("WEAR_SMOKE", "0") == "1"
if LOCAL:
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_k, "3")
import numpy as np, pandas as pd
import torch, torch.nn as nn
if LOCAL:
    torch.set_num_threads(3)

T0 = time.time()
def elapsed_min(): return (time.time() - T0) / 60.0
def log(*a): print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

ROOT = "/kaggle/input/competitions/3rd-wear-dataset-challenge-hasca-2026"
OUT = "/kaggle/working"
if LOCAL:
    DATA = r"E:\Claude code\wear\data"; PREP = os.path.join(DATA, "prep")
    OUT = os.path.join(r"E:\Claude code\wear\kaggle\uec_k2", "smoke_out" if SMOKE else "local_out")
os.makedirs(OUT, exist_ok=True)

NC = 19; WIN = 50; IMU_HZ = 50; VIDEO_HZ = 30; VWIN = 15
STRIDE = 50 if LOCAL else 25            # repo: --stride 25; the local prep only holds tile-aligned (stride 50) windows
SEL = [0, VWIN // 2, VWIN - 1]          # --video-frame-selection first_mid_last -> frames 0, 7, 14 of the 15-frame crop
SENSOR_KEYS = ["ra", "rl", "ll", "la"]  # == UEC SENSOR_COLS order == --sensor-keys ra rl ll la (one-hot order)
SENSOR_COLS = {"ra": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"],
               "rl": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
               "ll": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
               "la": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"]}
LOC2KI = {"right_arm": 0, "right_leg": 1, "left_leg": 2, "left_arm": 3}   # test sensor_location -> SENSOR_KEYS index
PREP_TO_UEC = [2, 3, 1, 0]              # prep limb order [la, ll, ra, rl] -> UEC order [ra, rl, ll, la]
CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)",
               "jogging (butt-kicks)", "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)",
               "stretching (hamstrings)", "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups",
               "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LAB2ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---- budget (minutes of wall time since script start)
GUARD_MIN = float(os.environ.get("WEAR_GUARD_MIN", "45"))                      # never START an epoch after this
DEADLINE = {"xcep": min(30.0, GUARD_MIN), "vmlp": min(38.0, GUARD_MIN), "vcnn": GUARD_MIN}   # per-member soft deadlines
EPOCHS = {"xcep": 22, "vmlp": 25, "vcnn": 25}
if SMOKE:
    EPOCHS = {k: 1 for k in EPOCHS}
EPOCHS = {k: int(os.environ.get(f"WEAR_EPOCHS_{k.upper()}", v)) for k, v in EPOCHS.items()}
MAX_BATCHES = int(os.environ.get("WEAR_MAX_BATCHES", "30" if SMOKE else "0")) or None
SEED = 42
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = dev.type == "cuda"
torch.backends.cudnn.benchmark = True
log(f"LOCAL={LOCAL} SMOKE={SMOKE} device={dev} amp={USE_AMP} epochs={EPOCHS} deadlines={DEADLINE} "
    f"guard={GUARD_MIN} max_batches={MAX_BATCHES} torch={torch.__version__}")


# ============================================================================ labels / windows (UEC rules)
def encode_labels(col):
    """UEC encode_label, vectorised: ''/'null' -> 0, class name -> id, 'nan'/'none' -> -1 (window skipped)."""
    s = col.astype(str).str.strip(); low = s.str.lower()
    out = np.full(len(s), -2, np.int64)
    for name, i in LAB2ID.items():
        out[(s == name).to_numpy()] = i
    out[((low == "") | (low == "null")).to_numpy()] = 0
    out[((low == "nan") | (low == "none")).to_numpy()] = -1
    bad = out == -2
    if bad.any():
        num = pd.to_numeric(s[bad], errors="coerce")
        out[bad] = np.where(num.notna().to_numpy(), num.fillna(-1).to_numpy().astype(np.float64).astype(np.int64), -1)
    return out


def window_labels(lab, starts):
    """assign_window_label_from_array(purity, 0.8): majority label if purity >= 0.8 and no None label, else -1."""
    L = lab[starts[:, None] + np.arange(WIN)[None, :]]
    bad = (L < 0).any(1)
    cnt = np.stack([(L == c).sum(1) for c in range(NC)], 1)
    y = cnt.argmax(1); pur = cnt.max(1) / float(WIN)
    return np.where((~bad) & (pur >= 0.8), y, -1).astype(np.int64), pur.astype(np.float32)


def finite_windows(fin_rows, starts):
    """fin_rows (n,) bool -> (W,) True where all WIN rows of the window are finite."""
    cs = np.concatenate([[0], np.cumsum(~fin_rows)])
    return (cs[starts + WIN] - cs[starts]) == 0


def crop_start(starts):
    """UEC center crop of the 30-fps video for a 50-sample window: 15 frames centred on the window (start=0 -> [8,23))."""
    return np.round((starts + WIN / 2.0) * VIDEO_HZ / IMU_HZ).astype(np.int64) - VWIN // 2


def finalize_stats(acc):
    """compute_normalization_stats: per-sensor mean/std over ALL rows of ALL training records (NaN -> 0)."""
    stats = {}
    for k in SENSOR_KEYS:
        s, ss, n = acc[k]
        m = s / n; var = np.maximum(ss / n - m ** 2, 1e-6); sd = np.sqrt(var)
        stats[k] = (m.astype(np.float32), np.where(sd < 1e-6, 1.0, sd).astype(np.float32))
    return stats


# ============================================================================ data loading
def load_kaggle():
    sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/train/inertial_feat/sbj_*.csv"))
    sessions = [s for s in sessions if not (len(s.split("_")) >= 3 and s.split("_")[-1] == "2")]   # --exclude-file-id-suffix-2
    log("train sessions", len(sessions), sessions)
    acc = {k: [np.zeros(3), np.zeros(3), 0] for k in SENSOR_KEYS}
    imu_w, y6, frames, y5 = [], [], [], []
    for s in sessions:
        df = pd.read_csv(f"{ROOT}/train/inertial_feat/{s}.csv", low_memory=False, keep_default_na=False)
        lab = encode_labels(df["label"]); n = len(df)
        A = np.stack([df[SENSOR_COLS[k]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32) for k in SENSOR_KEYS], 1)  # (n,4,3)
        for ki, k in enumerate(SENSOR_KEYS):
            v = np.nan_to_num(A[:, ki].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            acc[k][0] += v.sum(0); acc[k][1] += (v ** 2).sum(0); acc[k][2] += n
        starts = np.arange(0, n - WIN + 1, STRIDE)
        y, pur = window_labels(lab, starts)
        fin = np.isfinite(A).all(2)                                            # (n,4)
        fin_w = np.stack([finite_windows(fin[:, ki], starts) for ki in range(4)], 1)   # (W,4)
        m6 = (y >= 0) & fin_w.all(1)                                           # has_only_finite_sensor_windows over ra rl ll la
        st6 = starts[m6]
        imu_w.append(A[st6[:, None] + np.arange(WIN)[None, :]])                # (W6,50,4,3)
        y6.append(y[m6])
        V = np.load(f"{ROOT}/train/videomae_feat/{s}.npy", mmap_mode="r")
        s0 = crop_start(starts)
        m5 = (y >= 0) & fin_w[:, 0] & (s0 >= 0) & (s0 + VWIN <= V.shape[0])    # video members: --sensor-keys ra -> ra finite
        idx = s0[m5][:, None] + np.asarray(SEL)[None, :]
        Vf = np.asarray(V, np.float32)
        frames.append(np.nan_to_num(Vf[idx]).astype(np.float16)); y5.append(y[m5]); del Vf
        log(f"{s}: samples={n} frames={V.shape[0]} windows={len(starts)} labelled={(y >= 0).sum()} imu_ok={m6.sum()} video_ok={m5.sum()}")
    stats = finalize_stats(acc)
    tm = pd.read_csv(f"{ROOT}/test/test_meta_data.csv")
    TI = np.load(f"{ROOT}/test/test_inertial_data.npy").astype(np.float32)     # (12234,50,3)
    Xv = np.load(f"{ROOT}/test/test_videomae_data.npy", mmap_mode="r")         # (12234,768,15)
    TV = np.concatenate([np.nan_to_num(np.asarray(Xv[i:i + 1024], np.float32)[:, :, SEL].transpose(0, 2, 1)).astype(np.float16)
                         for i in range(0, Xv.shape[0], 1024)])                # (12234,3,768)
    return dict(imu_w=np.concatenate(imu_w), y6=np.concatenate(y6), frames=np.concatenate(frames), y5=np.concatenate(y5),
                stats=stats, tm=tm, TI=TI, TV=TV)


def load_local():
    meta = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
    IMU = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r")          # (N,4,50,3) f16, limbs la, ll, ra, rl
    VID = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")      # (N,15,160) f16 stand-in for raw 768-d frames
    keep = meta.session.str.count("_").to_numpy() < 2                          # exclude sbj_0_2, sbj_14_2
    if SMOKE:
        keep &= meta.session.isin(["sbj_0", "sbj_5", "sbj_20"]).to_numpy()
    rows = np.where(keep)[0]
    I = np.asarray(IMU[rows], np.float32)[:, PREP_TO_UEC]                      # (N,4,50,3) in UEC sensor order
    y = np.where(meta.pur.to_numpy()[rows] >= 0.8, meta.y.to_numpy()[rows].astype(np.int64), -1)
    acc = {}
    for ki, k in enumerate(SENSOR_KEYS):
        v = np.nan_to_num(I[:, ki].reshape(-1, 3).astype(np.float64))
        acc[k] = [v.sum(0), (v ** 2).sum(0), len(v)]
    stats = finalize_stats(acc)
    fin = np.isfinite(I).all(axis=(2, 3))                                      # (N,4)
    m6 = (y >= 0) & fin.all(1)
    m5 = (y >= 0) & fin[:, 0]
    frames = np.nan_to_num(np.asarray(VID[rows[m5]], np.float32)[:, SEL]).astype(np.float16)
    log(f"local sessions={sorted(set(meta.session[keep]))} tiles={len(rows)} labelled={(y >= 0).sum()} imu_ok={m6.sum()} video_ok={m5.sum()}")
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
    TI = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float32)
    TV = np.nan_to_num(np.asarray(np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r")[:, SEL], np.float32)).astype(np.float16)
    return dict(imu_w=I[m6].transpose(0, 2, 1, 3), y6=y[m6], frames=frames, y5=y[m5], stats=stats, tm=tm, TI=TI, TV=TV)


# ============================================================================ M6 input features (build_sensor_input + one-hot)
def lagdiff(W, lag):
    d = np.zeros_like(W); d[:, :, lag:] = W[:, :, lag:] - W[:, :, :-lag]; return d


def build_input(W, ki, stats):
    """W (M,50,3) raw acc of sensor ki (int or (M,) array) -> (M,20,50) float32:
    z-scored xyz, |acc|, diff1 xyz, |diff1|, diff5 xyz, |diff5|, diff10 xyz, |diff10|, one-hot(4) of the sensor."""
    W = np.nan_to_num(np.asarray(W, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    M = np.stack([stats[k][0] for k in SENSOR_KEYS]); S = np.stack([stats[k][1] for k in SENSOR_KEYS])   # (4,3)
    ki = np.broadcast_to(np.asarray(ki, np.int64), (len(W),))
    Wt = ((W - M[ki][:, None, :]) / S[ki][:, None, :]).astype(np.float32).transpose(0, 2, 1)             # (M,3,50)
    Wt = np.nan_to_num(Wt)
    d1 = np.diff(Wt, axis=2, prepend=Wt[:, :, :1]); d5 = lagdiff(Wt, 5); d10 = lagdiff(Wt, 10)
    nrm = lambda a: np.linalg.norm(a, axis=1, keepdims=True)
    oh = np.zeros((len(W), 4, W.shape[1]), np.float32); oh[np.arange(len(W)), ki] = 1.0
    X = np.concatenate([Wt, nrm(Wt), d1, nrm(d1), d5, nrm(d5), d10, nrm(d10), oh], 1)
    return np.nan_to_num(X).astype(np.float32)


# ============================================================================ video features (video_only_common reconstruction, see vport.py)
def delta_concat(v):
    """(B,T,D) -> (B,T,2D): [x, diff(x) with a zero first row]  (--video-feature-mode delta_concat)."""
    d = np.zeros_like(v); d[:, 1:] = v[:, 1:] - v[:, :-1]
    return np.concatenate([v, d], axis=-1)


def _rowcos(a, b, eps=1e-6):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps)


def _stats1d(x):
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    mn, mx = x.min(1), x.max(1)
    return np.stack([x.mean(1), x.std(1), mn, mx, mx - mn, q[0], q[1], q[2], q[3], q[4], q[3] - q[1],
                     np.sqrt((x ** 2).mean(1)), (x ** 2).sum(1), x[:, 0], x[:, -1], x[:, -1] - x[:, 0]], 1)


def mlp_feature_vector(v):
    """v (B,T,D) -> agg (B,3D) [mean_t, std_t, mean(last k)-mean(first k)], scalar (B,56); k = max(1, T//3)."""
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


def build_mlp_features(frames, chunk=8192):
    aggs, scs = [], []
    for s in range(0, len(frames), chunk):
        a, c = mlp_feature_vector(delta_concat(np.asarray(frames[s:s + chunk], np.float32)))
        aggs.append(a.astype(np.float16)); scs.append(c)
    return np.concatenate(aggs), np.concatenate(scs)


def delta_concat_t(F):
    """torch version for the CNN input: F (B,T,D) -> (B,T,2D)."""
    d = torch.zeros_like(F); d[:, 1:] = F[:, 1:] - F[:, :-1]
    return torch.cat([F, d], -1)


# ============================================================================ models
class ConvBN(nn.Sequential):            # tsai ConvBlock(ni, nf, 1[, act=None]): conv(bias=False) -> BN -> ReLU
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
        ks = [k if k % 2 != 0 else k - 1 for k in ks]        # 41 -> [41, 19, 9]
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
    """tsai XceptionTime (the repo's models/xceptiontime.py is not public): block -> AdaptiveAvgPool(adaptive_size)
    -> ConvBN(32nf,16nf) -> ConvBN(16nf,8nf) -> ConvBN(8nf,c_out) -> GAP.  Input (B, C, L)."""
    def __init__(self, c_in, c_out, nf=16, adaptive_size=50, residual=True, bottleneck=True, ks=40):
        super().__init__()
        self.block = XceptionBlock(c_in, nf, residual=residual, ks=ks, bottleneck=bottleneck)
        hn = nf * 32
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size), ConvBN(hn, hn // 2), ConvBN(hn // 2, hn // 4),
                                  ConvBN(hn // 4, c_out), nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x):
        return self.head(self.block(x))


class VideoMLP(nn.Module):              # train_video_mlp.VideoMLP defaults: hidden 1024, dropout 0.3
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, NC))

    def forward(self, x):
        return self.head(x)


class VideoCNN(nn.Module):              # train_video_cnn.VideoCNN defaults: proj 256, channels 256, hidden 256, dropout 0.2
    def __init__(self, input_feature_dim, proj_dim=256, channels=256, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_feature_dim, proj_dim, kernel_size=1)
        self.encoder = nn.Sequential(nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
                                     nn.Conv1d(proj_dim, channels, kernel_size=3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
                                     nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(channels, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, NC))

    def forward(self, video):           # (B,T,C)
        x = self.input_proj(video.transpose(1, 2))
        return self.head(self.encoder(x).squeeze(-1))


# ============================================================================ training
def autocast_ctx():
    if not USE_AMP:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except Exception:
        return torch.cuda.amp.autocast()


def make_scaler():
    if not USE_AMP:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


@torch.no_grad()
def predict(model, X, bs=2048):
    model.eval(); out = []
    with autocast_ctx():
        for s in range(0, len(X), bs):
            out.append(torch.softmax(model(X[s:s + bs].float()).float(), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def balanced_weights(y):                # train_video_mlp balanced_class_weights
    counts = np.bincount(y, minlength=NC).astype(np.float64); nz = counts > 0
    w = np.ones(NC, np.float64); w[nz] = len(y) / (float(nz.sum()) * counts[nz])
    return torch.tensor(w, dtype=torch.float32, device=dev)


TEST_IDS = None


def save_probs(P, name):
    """P rows follow test_meta rows; write in test-id order (ids are a permutation of 0..N-1)."""
    Q = np.empty_like(P); Q[TEST_IDS] = P
    np.save(os.path.join(OUT, name), Q.astype(np.float32))


def train_member(tag, model, X, y, XT, out_name, *, epochs, batch_size, lr, wd, class_weight, deadline_min, log_every=1000):
    """Fixed-epoch full-data fit, AdamW, per-step cosine LR (lr -> 0 at the last planned step).  Test probabilities
    are re-saved after EVERY epoch.  Adaptive plan: after epoch 1 the planned epoch count is reduced if the member
    would overrun its deadline, so the cosine schedule still finishes at the last epoch actually run."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model.to(dev)
    n = len(y); y_t = torch.as_tensor(np.asarray(y, np.int64), dtype=torch.long, device=dev)
    crit = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = make_scaler()
    g = torch.Generator().manual_seed(SEED)
    spe = math.ceil(n / batch_size)
    if MAX_BATCHES:
        spe = min(spe, MAX_BATCHES)
    planned = epochs; total_steps = max(planned * spe, 1); step = 0; hist = []; lr_now = lr
    log(f"[{tag}] samples={n} input={tuple(X.shape[1:])} params={sum(p.numel() for p in model.parameters())} "
        f"steps/epoch={spe} batch={batch_size} planned_epochs={planned} class_weight={'balanced' if class_weight is not None else 'none'}")
    for ep in range(1, epochs + 1):
        if ep > planned:
            break
        now = elapsed_min()
        if now >= GUARD_MIN or now >= deadline_min:
            log(f"[{tag}] TIME GUARD: not starting epoch {ep} (elapsed {now:.1f} min, member deadline {deadline_min:.0f}, guard {GUARD_MIN:.0f})")
            break
        te = time.time(); model.train(); tl = torch.zeros((), device=dev); nb = 0
        perm = torch.randperm(n, generator=g).to(dev)
        for bi in range(spe):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) < 2:
                continue
            lr_now = lr * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            xb = X[idx].float(); yb = y_t[idx]
            with autocast_ctx():
                logits = model(xb)
            loss = crit(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tl += loss.detach(); nb += 1; step += 1
            if log_every and nb % log_every == 0:
                log(f"[{tag}] ep{ep} batch {nb}/{spe} loss {float(tl) / nb:.4f} lr {lr_now:.2e}")
        t_train = time.time() - te
        P = predict(model, XT); save_probs(P, out_name)
        t_ep = time.time() - te
        rec = {"epoch": ep, "train_loss": float(tl) / max(nb, 1), "lr_end": lr_now, "train_sec": round(t_train, 1),
               "epoch_sec": round(t_ep, 1), "elapsed_min": round(elapsed_min(), 2), "pred_dist": np.bincount(P.argmax(1), minlength=NC).tolist()}
        hist.append(rec)
        json.dump(hist, open(os.path.join(OUT, f"hist_{tag}.json"), "w"), indent=1)
        log(f"[{tag}] epoch {ep}/{planned} loss {rec['train_loss']:.4f} lr {lr_now:.2e} epoch_time {t_ep:.1f}s "
            f"(train {t_train:.1f}s) elapsed {rec['elapsed_min']:.1f} min -> saved {out_name}")
        if ep == 1 and planned > 1:
            fit = 1 + int(max(0.0, deadline_min - elapsed_min()) * 60.0 / max(t_ep, 1e-6))
            if fit < planned:
                planned = max(fit, 1); total_steps = max(planned * spe, 1)
                log(f"[{tag}] ADAPTIVE PLAN: {epochs} epochs would overrun the {deadline_min:.0f}-min deadline "
                    f"({t_ep:.0f}s/epoch); planned_epochs -> {planned}")
    torch.save(model.state_dict(), os.path.join(OUT, f"{tag}_model.pt"))
    return hist, planned


def main():
    global TEST_IDS
    D = load_local() if LOCAL else load_kaggle()
    tm = D["tm"]; TEST_IDS = tm["id"].to_numpy().astype(np.int64)
    assert len(TEST_IDS) == len(D["TI"]) == len(D["TV"]) and sorted(TEST_IDS.tolist()) == list(range(len(TEST_IDS))), "test ids are not 0..N-1"
    stats = D["stats"]
    log("norm stats", {k: (stats[k][0].round(3).tolist(), stats[k][1].round(3).tolist()) for k in SENSOR_KEYS})
    log(f"M6 windows={len(D['y6'])} (x4 sensors) class_counts={np.bincount(D['y6'], minlength=NC).tolist()}")
    log(f"video windows={len(D['y5'])} frame_dim={D['frames'].shape[-1]} class_counts={np.bincount(D['y5'], minlength=NC).tolist()}")
    summary = {"local": LOCAL, "smoke": SMOKE, "device": str(dev), "amp": USE_AMP, "stride": STRIDE, "epochs_planned": EPOCHS,
               "n_imu_windows": int(len(D["y6"])), "n_video_windows": int(len(D["y5"])), "frame_dim": int(D["frames"].shape[-1])}

    # ------------------------------------------------------------------ M6 Inertial-XceptionTime
    X6 = np.concatenate([build_input(D["imu_w"][:, :, ki, :], ki, stats) for ki in range(4)], 0)      # (4*W6,20,50)
    Y6 = np.tile(D["y6"], 4)
    kk = tm["sensor_location"].map(LOC2KI).to_numpy().astype(np.int64)
    XT6 = build_input(D["TI"], kk, stats)                                                              # row's own sensor
    log(f"M6 features built: X {X6.shape} XT {XT6.shape} ({elapsed_min():.1f} min)")
    X6_t = torch.from_numpy(X6).to(dev); XT6_t = torch.from_numpy(XT6).to(dev); del X6
    model6 = XceptionTime(c_in=20, c_out=NC, nf=48, adaptive_size=8, residual=True, bottleneck=True, ks=41)
    hist6, plan6 = train_member("xcep", model6, X6_t, Y6, XT6_t, "test_xcep.npy", epochs=EPOCHS["xcep"], batch_size=128,
                                lr=1e-3, wd=1e-4, class_weight=None, deadline_min=DEADLINE["xcep"], log_every=1000)
    summary["xcep"] = {"epochs_done": len(hist6), "epochs_planned_final": plan6, "n_samples": int(len(Y6))}
    del X6_t, XT6_t, model6, D["imu_w"]
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ video features (shared by M5 / M4)
    frames, TV = D["frames"], D["TV"]
    agg, sc = build_mlp_features(frames); tagg, tsc = build_mlp_features(TV)
    mu = sc.mean(0); sd = np.maximum(sc.std(0), 1e-6)                        # scalar block z-scored with train statistics
    X5 = np.concatenate([agg, ((sc - mu) / sd).astype(np.float16)], 1); XT5 = np.concatenate([tagg, ((tsc - mu) / sd).astype(np.float16)], 1)
    log(f"M5 features built: X {X5.shape} XT {XT5.shape} ({elapsed_min():.1f} min)")
    del agg, sc, tagg, tsc

    # ------------------------------------------------------------------ M5 Video-MLP
    X5_t = torch.from_numpy(X5).to(dev); XT5_t = torch.from_numpy(XT5).to(dev); del X5, XT5
    model5 = VideoMLP(X5_t.shape[1])
    hist5, plan5 = train_member("vmlp", model5, X5_t, D["y5"], XT5_t, "test_vmlp.npy", epochs=EPOCHS["vmlp"], batch_size=256,
                                lr=1e-3, wd=1e-4, class_weight=balanced_weights(D["y5"]), deadline_min=DEADLINE["vmlp"], log_every=0)
    summary["vmlp"] = {"epochs_done": len(hist5), "epochs_planned_final": plan5, "n_samples": int(len(D["y5"])), "input_dim": int(X5_t.shape[1])}
    del X5_t, XT5_t, model5
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ M4 Video-CNN
    X4_t = delta_concat_t(torch.from_numpy(frames).to(dev)); XT4_t = delta_concat_t(torch.from_numpy(TV).to(dev))   # (B,3,2D) f16
    model4 = VideoCNN(X4_t.shape[2])
    hist4, plan4 = train_member("vcnn", model4, X4_t, D["y5"], XT4_t, "test_vcnn.npy", epochs=EPOCHS["vcnn"], batch_size=256,
                                lr=1e-3, wd=1e-4, class_weight=balanced_weights(D["y5"]), deadline_min=DEADLINE["vcnn"], log_every=0)
    summary["vcnn"] = {"epochs_done": len(hist4), "epochs_planned_final": plan4, "n_samples": int(len(D["y5"])), "input_shape": list(X4_t.shape[1:])}

    # ------------------------------------------------------------------ outputs
    for name in ("test_xcep.npy", "test_vmlp.npy", "test_vcnn.npy"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            P = np.load(p)
            ok = P.shape == (len(TEST_IDS), NC) and np.isfinite(P).all() and np.allclose(P.sum(1), 1.0, atol=1e-3)
            summary[name] = {"shape": list(P.shape), "dtype": str(P.dtype), "valid": bool(ok), "pred_dist": np.bincount(P.argmax(1), minlength=NC).tolist()}
            log(name, P.shape, P.dtype, "valid" if ok else "INVALID", "pred_dist", summary[name]["pred_dist"])
        else:
            summary[name] = None; log(name, "MISSING")
    summary["total_min"] = round(elapsed_min(), 2)
    json.dump(summary, open(os.path.join(OUT, "k2_summary.json"), "w"), indent=1)
    log(f"DONE in {elapsed_min():.1f} min; outputs in {OUT}: {sorted(os.listdir(OUT))}")


if __name__ == "__main__":
    main()
