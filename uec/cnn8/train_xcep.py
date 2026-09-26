"""CPU port of UEC-dx2 approach_XceptionTime/src/train_inertial_xceptiontime.py (ensemble input 6, run
reproduce_20260623_095214_xceptiontime_cv5):
  --model xceptiontime --sensor-keys ra rl ll la --window-size 50 --stride 25 --purity 0.8
  --normalization-mode global --add-magnitude --add-diff --add-diff-magnitude --add-diff-5 --add-diff-5-magnitude
  --add-diff-10 --add-diff-10-magnitude --channels 48 --kernel-size 41 --adaptive-size 8 --epochs 80
  --batch-size 128 --learning-rate 0.001 --weight-decay 0.0001 --dropout 0.0 --early-stopping-rounds 8
  --class-weight none --seed 42
The repo's XceptionTime class (approach_XceptionTime/src/models/xceptiontime.py) is not in the public repo; it
is re-implemented here from tsai's XceptionTime (same signature: c_in, c_out, nf, adaptive_size, residual,
bottleneck, ks), in pure torch.  CPU budget forces far fewer epochs than 80 (see --epochs / --max-minutes).
"""
import os, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, pandas as pd
import torch, torch.nn as nn
torch.set_num_threads(4)
from sklearn.metrics import f1_score, accuracy_score

HERE = r"E:\Claude code\wear\uec\cnn8"; CACHE = os.path.join(HERE, "cache"); DATA = r"E:\Claude code\wear\data"
NC = 19
SENSOR_ORDER = ["ra", "rl", "ll", "la"]            # cache order == UEC --sensor-keys ra rl ll la
SENSOR_COLS = {"ra": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"], "rl": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
               "ll": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"], "la": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"]}
LIMB_TO_K = [3, 2, 0, 1]

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["val4", "full", "std"], required=True)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--epochs", type=int, default=80)
ap.add_argument("--max-minutes", type=float, default=1e9, help="stop starting new epochs after this wall time")
ap.add_argument("--lr-replay", type=str, default=None)
ap.add_argument("--channels", type=int, default=48)
ap.add_argument("--kernel-size", type=int, default=41)
ap.add_argument("--adaptive-size", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--learning-rate", type=float, default=1e-3)
ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--early-stopping-rounds", type=int, default=8)
ap.add_argument("--scheduler-patience", type=int, default=3)
ap.add_argument("--scheduler-factor", type=float, default=0.5)
ap.add_argument("--train-frac", type=float, default=1.0, help="random fraction of training tiles per epoch (1.0 = all)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", type=str, required=True)
args = ap.parse_args()
OUT = os.path.join(HERE, "runs", args.tag); os.makedirs(OUT, exist_ok=True)
logf = open(os.path.join(OUT, "train.log"), "a"); t0 = time.time()
def log(*a):
    s = f"[{time.time()-t0:7.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
np.random.seed(args.seed); torch.manual_seed(args.seed)

# ---------------- tsai XceptionTime (pure torch)
class ConvBN(nn.Sequential):   # tsai ConvBlock(ni, nf, 1[, act=None])
    def __init__(self, ni, nf, act=True):
        layers = [nn.Conv1d(ni, nf, 1, bias=False), nn.BatchNorm1d(nf)]
        if act: layers.append(nn.ReLU())
        super().__init__(*layers)

class SeparableConv1d(nn.Module):
    def __init__(self, ni, nf, ks):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(ni, ni, ks, padding=ks // 2, groups=ni, bias=False)
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, bias=False)
    def forward(self, x): return self.pointwise_conv(self.depthwise_conv(x))

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
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size), ConvBN(hn, hn // 2), ConvBN(hn // 2, hn // 4), ConvBN(hn // 4, c_out),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten())
    def forward(self, x): return self.head(self.block(x))

# ---------------- data
meta = pd.read_csv(os.path.join(DATA, "prep", "train_meta.csv")); sbj = meta.sbj.to_numpy()
imu = np.load(os.path.join(CACHE, "imu.npy"), mmap_mode="r")
y_uec = np.load(os.path.join(CACHE, "y_uec.npy")); suffix2 = np.load(os.path.join(CACHE, "suffix2.npy"))
fin4 = np.isfinite(np.asarray(imu)).all(axis=(2, 3))
ok = (y_uec >= 0) & fin4.all(1)
perm = np.random.RandomState(0).permutation(np.unique(sbj)); fo = {s: i % 5 for i, s in enumerate(perm)}; std_fold = np.array([fo[s] for s in sbj])
val_s = [18, 19, 20, 21]
if args.mode == "val4":
    tr = np.where(ok & ~suffix2 & ~np.isin(sbj, val_s))[0]; va = np.where(ok & ~suffix2 & np.isin(sbj, val_s))[0]
    train_sessions = sorted(set(meta.session[~suffix2 & ~np.isin(sbj, val_s)]))
elif args.mode == "full":
    tr = np.where(ok & ~suffix2)[0]; va = np.zeros(0, np.int64); train_sessions = sorted(set(meta.session[~suffix2]))
else:
    tr = np.where(ok & ~suffix2 & (std_fold != args.fold))[0]; va = np.zeros(0, np.int64)
    train_sessions = sorted(set(meta.session[~suffix2 & (std_fold != args.fold)]))

# global normalisation stats per sensor over the FULL training records (NaN -> 0), UEC compute_normalization_stats
stats = {}
acc = {k: [np.zeros(3), np.zeros(3), 0] for k in SENSOR_ORDER}
for s in train_sessions:
    df = pd.read_csv(os.path.join(DATA, "train", "inertial_feat", f"{s}.csv"), low_memory=False, keep_default_na=False)
    for k in SENSOR_ORDER:
        v = np.nan_to_num(df[SENSOR_COLS[k]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        acc[k][0] += v.sum(0); acc[k][1] += (v ** 2).sum(0); acc[k][2] += len(v)
for k in SENSOR_ORDER:
    m = acc[k][0] / acc[k][2]; var = np.maximum(acc[k][1] / acc[k][2] - m ** 2, 0.0); sd = np.sqrt(var)
    stats[k] = (m.astype(np.float32), np.where(sd < 1e-6, 1.0, sd).astype(np.float32))
log("norm stats", {k: (stats[k][0].round(3).tolist(), stats[k][1].round(3).tolist()) for k in SENSOR_ORDER})

def lagdiff(W, lag):  # W (M,3,L)
    d = np.zeros_like(W); d[:, :, lag:] = W[:, :, lag:] - W[:, :, :-lag]; return d
def build_input(W, k):  # W (M,50,3) raw -> (M,20,50)
    W = np.nan_to_num(W.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    m, sd = stats[SENSOR_ORDER[k]]
    Wt = ((W - m) / sd).astype(np.float32).transpose(0, 2, 1)       # (M,3,50)
    d1 = np.diff(Wt, axis=2, prepend=Wt[:, :, :1]); d5 = lagdiff(Wt, 5); d10 = lagdiff(Wt, 10)
    nrm = lambda a: np.linalg.norm(a, axis=1, keepdims=True)
    oh = np.zeros((len(W), 4, W.shape[1]), np.float32); oh[:, k] = 1.0
    X = np.concatenate([Wt, nrm(Wt), d1, nrm(d1), d5, nrm(d5), d10, nrm(d10), oh], 1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
def build(tiles):
    I = np.asarray(imu[tiles])
    X = np.concatenate([build_input(I[:, k], k) for k in range(4)], 0)
    return torch.from_numpy(X), np.concatenate([y_uec[tiles]] * 4)
Xtr, ytr = build(tr); ytr_t = torch.from_numpy(ytr)
if len(va): Xva, yva = build(va)
log(f"mode={args.mode} train samples={len(ytr)} val samples={len(va)*4} input={tuple(Xtr.shape[1:])}")

model = XceptionTime(c_in=20, c_out=NC, nf=args.channels, adaptive_size=args.adaptive_size, residual=True, bottleneck=True, ks=args.kernel_size)
log("params", sum(p.numel() for p in model.parameters()))
crit = nn.CrossEntropyLoss()   # class_weight none
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience)
replay = None
if args.lr_replay:
    replay = [args.learning_rate] + pd.read_csv(args.lr_replay)["lr"].tolist()

@torch.inference_mode()
def predict(X, bs=2048):
    model.eval(); return torch.cat([torch.softmax(model(X[b:b + bs]), 1) for b in range(0, len(X), bs)]).numpy()

best, best_ep, bad, hist, best_state = -1.0, None, 0, [], None
g = torch.Generator().manual_seed(args.seed); n = len(ytr)
for ep in range(1, args.epochs + 1):
    if (time.time() - t0) / 60 > args.max_minutes:
        log("time budget reached; stop before epoch", ep); break
    if replay is not None:
        for pg in opt.param_groups: pg["lr"] = replay[min(ep - 1, len(replay) - 1)]
    lr_used = opt.param_groups[0]["lr"]
    model.train(); tl = nb = 0; te = time.time()
    order = torch.randperm(n, generator=g)
    if args.train_frac < 1.0: order = order[:int(n * args.train_frac)]
    for bi, b in enumerate(range(0, len(order), args.batch_size)):
        idx = order[b:b + args.batch_size]
        if len(idx) < 2: continue
        loss = crit(model(Xtr[idx]), ytr_t[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl += float(loss.item()); nb += 1
        if bi % 200 == 0: log(f"  ep{ep} batch {bi}/{(len(order)+args.batch_size-1)//args.batch_size} loss {tl/nb:.4f}")
    row = {"epoch": ep, "train_loss": tl / max(nb, 1), "lr_used": lr_used, "sec": time.time() - te}
    if len(va):
        P = predict(Xva); f = f1_score(yva, P.argmax(1), average="macro", zero_division=0)
        sched.step(f); row.update({"val_macro_f1": f, "val_acc": accuracy_score(yva, P.argmax(1))})
        if f > best:
            best, best_ep, bad = f, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "best_epoch": ep}, os.path.join(OUT, "best.pt"))
            np.save(os.path.join(OUT, "val_probabilities.npy"), P.astype(np.float32))
        else:
            bad += 1
    row["lr"] = opt.param_groups[0]["lr"]; hist.append(row); pd.DataFrame(hist).to_csv(os.path.join(OUT, "history.csv"), index=False)
    log(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if len(va) and bad >= args.early_stopping_rounds: log("early stopping"); break
if best_state is not None:
    model.load_state_dict(best_state); log(f"best epoch={best_ep} val_macro_f1={best:.6f}")
torch.save({"model_state_dict": model.state_dict()}, os.path.join(OUT, "final.pt"))

tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv")); assert (tm.id.to_numpy() == np.arange(len(tm))).all()
TI = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float32)
KEYK = {"right_arm": 0, "right_leg": 1, "left_leg": 2, "left_arm": 3}
kk = tm.sensor_location.map(KEYK).to_numpy()
Pte = np.zeros((len(tm), NC), np.float32)
for k in range(4):
    sel = np.where(kk == k)[0]
    Pte[sel] = predict(torch.from_numpy(build_input(TI[sel], k)))
np.save(os.path.join(OUT, "test_probabilities.npy"), Pte)
log("test saved", Pte.shape, "pred dist", np.bincount(Pte.argmax(1), minlength=NC).tolist())
if args.mode == "std":
    rows = np.where(std_fold == args.fold)[0]; I = np.asarray(imu[rows])
    oof = np.full((len(rows), 4, NC), np.nan, np.float32)
    for li, k in enumerate(LIMB_TO_K):
        valid = np.isfinite(I[:, k]).all(axis=(1, 2))
        if valid.any(): oof[valid, li] = predict(torch.from_numpy(build_input(I[valid][:, k], k)))
    np.save(os.path.join(OUT, "oof_rows.npy"), rows); np.save(os.path.join(OUT, "oof_fold.npy"), oof); log("oof saved")
log("DONE")
