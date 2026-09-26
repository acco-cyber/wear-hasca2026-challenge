"""FULL-DATA multi-seed refit of the multimodal fusion net (IMU Conv1D + VideoMAE-frame Transformer), subject-disjoint 5-fold, test-format windows.
Runs on Kaggle (full 768-d frames) or locally in smoke mode (PCA-160 prep artifacts).
Outputs: oof.npy (N,4,19) float32 (NaN where limb missing/unused), test.npy (12234,19), meta csv, cv.json
"""
import os, sys, glob, time, json, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fnn
from sklearn.metrics import f1_score

LOCAL = os.environ.get("WEAR_LOCAL", "0") == "1"
SMOKE = os.environ.get("WEAR_SMOKE", "0") == "1"
ROOT = "/kaggle/input/competitions/3rd-wear-dataset-challenge-hasca-2026"
OUT = "/kaggle/working" if not LOCAL else r"E:\Claude code\wear\work\fusion_local"
PREP = r"E:\Claude code\wear\data\prep"
os.makedirs(OUT, exist_ok=True)
EPOCHS = int(os.environ.get("WEAR_EPOCHS", "2" if SMOKE else "10"))
FOLDS = 5; NC = 19; LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
FS, FPS, V0, V1 = 50, 30, 8, 23
CLASS_NAMES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)", "jogging (butt-kicks)",
    "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)", "stretching (lumbar rotation)",
    "push-ups", "push-ups (complex)", "sit-ups", "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LAB2ID = {c: i for i, c in enumerate(CLASS_NAMES)}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ data
if LOCAL:
    meta = pd.read_csv(f"{PREP}/train_meta.csv")
    IMU = np.load(f"{PREP}/train_imu.npy")                    # (N,4,50,3) f16
    VID = np.load(f"{PREP}/train_vid_pca.npy")                # (N,15,160) f16
    tm = pd.read_csv(r"E:\Claude code\wear\data\test\test_meta_data.csv")
    XI_te = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy").astype(np.float32)
    VID_te = np.load(f"{PREP}/test_vid_pca.npy")
    if SMOKE:
        keep = meta.session.isin(["sbj_0", "sbj_5", "sbj_10", "sbj_20", "sbj_21"]).to_numpy()
        meta = meta[keep].reset_index(drop=True); IMU = IMU[keep]; VID = VID[keep]
else:
    sessions = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/train/inertial_feat/sbj_*.csv"))
    metas, imus, vids = [], [], []
    for s in sessions:
        df = pd.read_csv(f"{ROOT}/train/inertial_feat/{s}.csv")
        A = df[[f"{l}_acc_{a}" for l in LIMBS for a in "xyz"]].to_numpy(np.float32)
        lab = df["label"].map(lambda v: LAB2ID.get(v, 0) if isinstance(v, str) else 0).to_numpy(np.int64)
        V = np.load(f"{ROOT}/train/videomae_feat/{s}.npy", mmap_mode="r")
        n = min(len(A) // FS, (V.shape[0] - V1) // FPS + 1)
        imu = A[: n * FS].reshape(n, FS, 4, 3).transpose(0, 2, 1, 3)
        L = lab[: n * FS].reshape(n, FS)
        y = np.array([np.bincount(r, minlength=NC).argmax() for r in L], np.int8); pur = (L == y[:, None]).mean(1)
        idx = (np.arange(n)[:, None] * FPS + np.arange(V0, V1)[None, :]).reshape(-1)
        vids.append(np.asarray(V[idx], np.float16).reshape(n, V1 - V0, -1)); imus.append(imu.astype(np.float16))
        metas.append(pd.DataFrame({"session": s, "sbj": int(s.split("_")[1]), "t": np.arange(n), "y": y, "pur": pur}))
        log(s, n)
    meta = pd.concat(metas, ignore_index=True); IMU = np.concatenate(imus); VID = np.concatenate(vids); del imus, vids
    tm = pd.read_csv(f"{ROOT}/test/test_meta_data.csv")
    XI_te = np.load(f"{ROOT}/test/test_inertial_data.npy").astype(np.float32)
    Xv = np.load(f"{ROOT}/test/test_videomae_data.npy", mmap_mode="r")
    VID_te = np.concatenate([np.asarray(Xv[i:i + 1024], np.float32).transpose(0, 2, 1).astype(np.float16) for i in range(0, Xv.shape[0], 1024)])
    meta.to_csv(f"{OUT}/train_meta.csv", index=False)
N = len(meta); DV = VID.shape[-1]
y_sec = meta.y.to_numpy().astype(np.int64); pur = meta.pur.to_numpy(); sbj = meta.sbj.to_numpy()
limb_te = np.array([LIMBS.index(l) for l in tm.sensor_location])
log("seconds", N, "video dim", DV, "test", len(tm), "device", dev)

# video standardisation stats (train frames)
vm = np.asarray(VID[::7], np.float32).reshape(-1, DV); v_mu = vm.mean(0); v_sd = vm.std(0) + 1e-6; del vm

# rows: (second, limb) with valid limb and purity >= 0.8
rows_sec, rows_limb = [], []
for li in range(4):
    ok = ~np.isnan(np.asarray(IMU[:, li], np.float32)).any(axis=(1, 2)) & (pur >= 0.8)
    rows_sec.append(np.where(ok)[0]); rows_limb.append(np.full(ok.sum(), li))
rows_sec = np.concatenate(rows_sec); rows_limb = np.concatenate(rows_limb)
log("rows", len(rows_sec))

class DS(torch.utils.data.Dataset):
    def __init__(self, secs, limbs, train):
        self.s, self.l, self.train = secs, limbs, train
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s, l = self.s[i], self.l[i]
        x = np.nan_to_num(np.asarray(IMU[s, l], np.float32))          # (50,3)
        v = np.nan_to_num((np.asarray(VID[s], np.float32) - v_mu) / v_sd)   # (15,DV)
        if self.train:
            x = x * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.03, x.shape).astype(np.float32)
            if np.random.rand() < 0.5:   # small random rotation about a random axis
                th = np.deg2rad(np.random.uniform(-15, 15)); ax = np.random.randint(3)
                c, s_ = math.cos(th), math.sin(th); R = np.eye(3, dtype=np.float32)
                i1, i2 = [k for k in range(3) if k != ax]; R[i1, i1] = c; R[i1, i2] = -s_; R[i2, i1] = s_; R[i2, i2] = c
                x = x @ R.T
        return torch.from_numpy(x), torch.from_numpy(v.astype(np.float32)), int(l), int(y_sec[s])

def imu_channels(x):   # (B,50,3) -> (B,7,50): raw, per-window z-scored, magnitude
    z = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-3)
    m = x.norm(dim=2, keepdim=True)
    return torch.cat([x, z, m], 2).transpose(1, 2)

class Net(nn.Module):
    def __init__(self, dv, d=192, p_mod=0.15):
        super().__init__()
        self.imu = nn.Sequential(
            nn.Conv1d(7, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=4, dilation=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), nn.BatchNorm1d(128), nn.GELU())
        self.limb = nn.Embedding(4, 16)
        self.imu_out = nn.Linear(256 + 16, d)
        self.vproj = nn.Sequential(nn.Linear(dv, d), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.pos = nn.Parameter(torch.zeros(1, 16, d))
        enc = nn.TransformerEncoderLayer(d, 4, d * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.vt = nn.TransformerEncoder(enc, 2)
        self.p_mod = p_mod
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, NC))
        self.head_imu = nn.Linear(d, NC); self.head_vid = nn.Linear(d, NC)
    def forward(self, x, v, l):
        h = self.imu(imu_channels(x)); h = torch.cat([h.mean(2), h.amax(2), self.limb(l)], 1); h = self.imu_out(h)
        z = self.vproj(v); z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos[:, : z.shape[1] + 1]
        z = self.vt(z)[:, 0]
        if self.training and self.p_mod > 0:
            keep_v = (torch.rand(len(z), 1, device=z.device) > self.p_mod).float(); keep_i = (torch.rand(len(h), 1, device=h.device) > self.p_mod).float()
            both_off = (keep_v == 0) & (keep_i == 0); keep_i = torch.where(both_off, torch.ones_like(keep_i), keep_i)
            z = z * keep_v; h = h * keep_i
        return self.head(torch.cat([h, z], 1)), self.head_imu(h), self.head_vid(z)

def predict(model, secs, limbs, imu_src=None, vid_src=None, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(secs), bs):
            s = secs[i:i + bs]; l = limbs[i:i + bs]
            if imu_src is None:
                x = np.asarray(IMU[s, l], np.float32); v = np.asarray(VID[s], np.float32)
            else:
                x = imu_src[s]; v = np.asarray(vid_src[s], np.float32)
            x = np.nan_to_num(x); v = np.nan_to_num((v - v_mu) / v_sd)
            lo, _, _ = model(torch.from_numpy(x).to(dev), torch.from_numpy(v.astype(np.float32)).to(dev), torch.from_numpy(l).to(dev))
            out.append(torch.softmax(lo.float(), 1).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ full-data multi-seed refit (GPU budget guard)
SEEDS = [0, 1, 2]; BUDGET_START_S = 40 * 60          # never START a new seed after 40 min of wall time
rng = np.random.RandomState(0)
cnt = np.bincount(y_sec[rows_sec], minlength=NC); keep_null = int(3 * np.median(cnt[1:]))
te_secs = np.arange(len(tm)); preds = []
for sd in SEEDS:
    if time.time() - t0 > BUDGET_START_S:
        log(f"time guard: skip seed {sd}"); break
    torch.manual_seed(sd); np.random.seed(sd); r = np.random.RandomState(sd)
    trs, trl = rows_sec.copy(), rows_limb.copy()
    nidx = np.where(y_sec[trs] == 0)[0]; sel = np.ones(len(trs), bool)
    if len(nidx) > keep_null: sel[r.choice(nidx, len(nidx) - keep_null, replace=False)] = False
    trs, trl = trs[sel], trl[sel]
    cnt2 = np.bincount(y_sec[trs], minlength=NC).astype(np.float64)
    w_cls = torch.tensor((cnt2.mean() / np.maximum(cnt2, 1)) ** 0.5, dtype=torch.float32, device=dev)
    dl = torch.utils.data.DataLoader(DS(trs, trl, True), batch_size=512, shuffle=True, num_workers=0 if LOCAL else 3, drop_last=True)
    model = Net(DV).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = EPOCHS * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    log(f"seed {sd}: train rows {len(trs)} (null kept {keep_null}), steps/epoch {len(dl)}")
    for ep in range(EPOCHS):
        model.train(); tl = 0; nb = 0
        for x, v, l, yy in dl:
            x, v, l, yy = x.to(dev), v.to(dev), l.to(dev), yy.to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                lo, li, lv = model(x, v, l)
                loss = Fnn.cross_entropy(lo, yy, weight=w_cls, label_smoothing=0.05) + 0.3 * (Fnn.cross_entropy(li, yy, weight=w_cls) + Fnn.cross_entropy(lv, yy, weight=w_cls))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        log(f"seed {sd} ep {ep}: loss {tl/nb:.4f}")
    p = predict(model, te_secs, limb_te, imu_src=XI_te, vid_src=VID_te).astype(np.float32)
    np.save(f"{OUT}/test_full_s{sd}.npy", p); preds.append(p)
    np.save(f"{OUT}/test.npy", np.mean(preds, 0).astype(np.float32))          # saved after every seed
    log(f"seed {sd} done; saved test.npy from {len(preds)} seed(s)")
log("DONE", len(preds), "seeds")
