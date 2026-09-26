"""CPU port of UEC-dx2 approach_base/src/train_exp024_cnn8_videomae_aux.py (ensemble input 2,
"Inertial-CNN+Video-MLP", run exp024_reg_mid_video_drop055_cv5):
  defaults + --weight-decay 2e-3 --classifier-dropout 0.50 --video-dropout 0.55 --cnn-dropout 0.25
  --label-smoothing 0.07 --early-stopping-rounds 5 --scheduler-patience 2 --scheduler-factor 0.5
Model / features / loss / optimiser / scheduler / early stopping are copied from the repo.  The torch
DataLoader is replaced by in-memory tensor batching (same math, no per-sample python).

Modes
  val4 : UEC's own holdout (train sbj 0-17, val sbj 18-21, both without *_2 sessions), early stopping on val
  full : all 22 subjects (no *_2), fixed epochs + LR schedule replayed from a val4 history
  std  : our standard protocol fold K (perm RandomState(0), i%5), fixed epochs + replayed LR; OOF for all 4 limbs
"""
import os, time, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
from sklearn.metrics import f1_score, accuracy_score

HERE = r"E:\Claude code\wear\uec\cnn8"; CACHE = os.path.join(HERE, "cache")
DATA = r"E:\Claude code\wear\data"
NC = 19; VIDEO_DIM = 768; VIDEO_SCALAR_DIM = 56
EMB_OF_K = np.array([0, 2, 3, 1])   # cache sensor order ra, rl, ll, la -> UEC SENSOR_TO_ID {ra:0, la:1, rl:2, ll:3}
LIMB_TO_K = [3, 2, 0, 1]            # our LIMBS (left_arm, left_leg, right_arm, right_leg) -> cache sensor index

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["val4", "full", "std"], required=True)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr-replay", type=str, default=None, help="history.csv of a val4 run; replays lr per epoch")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", type=str, required=True)
# repo hyper-parameters (member 2)
ap.add_argument("--cnn-base-channels", type=int, default=64)
ap.add_argument("--cnn-dropout", type=float, default=0.25)
ap.add_argument("--video-hidden-dim", type=int, default=128)
ap.add_argument("--video-out-dim", type=int, default=128)
ap.add_argument("--video-dropout", type=float, default=0.55)
ap.add_argument("--classifier-hidden", type=int, default=256)
ap.add_argument("--classifier-dropout", type=float, default=0.50)
ap.add_argument("--sensor-emb-dim", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=1024)
ap.add_argument("--learning-rate", type=float, default=5e-4)
ap.add_argument("--weight-decay", type=float, default=2e-3)
ap.add_argument("--early-stopping-rounds", type=int, default=5)
ap.add_argument("--label-smoothing", type=float, default=0.07)
ap.add_argument("--scheduler-patience", type=int, default=2)
ap.add_argument("--scheduler-factor", type=float, default=0.5)
args = ap.parse_args()

OUT = os.path.join(HERE, "runs", args.tag); os.makedirs(OUT, exist_ok=True)
logf = open(os.path.join(OUT, "train.log"), "a")
t0 = time.time()
def log(*a):
    s = f"[{time.time()-t0:7.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

np.random.seed(args.seed); torch.manual_seed(args.seed)

# ---------------- data
meta = pd.read_csv(os.path.join(DATA, "prep", "train_meta.csv"))
sbj = meta.sbj.to_numpy()
imu = np.load(os.path.join(CACHE, "imu.npy"), mmap_mode="r")
y_uec = np.load(os.path.join(CACHE, "y_uec.npy")); train_ok = np.load(os.path.join(CACHE, "train_ok.npy"))
suffix2 = np.load(os.path.join(CACHE, "suffix2.npy"))
VM = torch.from_numpy(np.load(os.path.join(CACHE, "vi_mean.npy")))
VS = torch.from_numpy(np.load(os.path.join(CACHE, "vi_std.npy")))
VD = torch.from_numpy(np.load(os.path.join(CACHE, "vi_d5.npy")))
SC_raw = np.load(os.path.join(CACHE, "scalar.npy"))

perm = np.random.RandomState(0).permutation(np.unique(sbj)); fo = {s: i % 5 for i, s in enumerate(perm)}
std_fold = np.array([fo[s] for s in sbj])

if args.mode == "val4":
    tr_tiles = np.where(train_ok & ~suffix2 & ~np.isin(sbj, [18, 19, 20, 21]))[0]
    va_tiles = np.where(train_ok & ~suffix2 & np.isin(sbj, [18, 19, 20, 21]))[0]
elif args.mode == "full":
    tr_tiles = np.where(train_ok & ~suffix2)[0]; va_tiles = np.zeros(0, np.int64)
else:
    tr_tiles = np.where(train_ok & ~suffix2 & (std_fold != args.fold))[0]; va_tiles = np.zeros(0, np.int64)
log(f"mode={args.mode} fold={args.fold} train_tiles={len(tr_tiles)} val_tiles={len(va_tiles)} "
    f"train_sbj={sorted(set(sbj[tr_tiles].tolist()))}")

def make_cnn8(W, eps=1e-6):  # W (M,50,3) -> (M,8,50)  (UEC make_cnn8_inertial)
    W = np.nan_to_num(np.asarray(W, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x, y, z = W[..., 0], W[..., 1], W[..., 2]
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2 + eps)
    d = np.diff(W, axis=1, prepend=W[:, :1]); ad = np.abs(d); dm = np.linalg.norm(d, axis=2)
    out = np.stack([x, y, z, mag, ad[..., 0], ad[..., 1], ad[..., 2], dm], 1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def build_samples(tiles):
    I = np.asarray(imu[np.sort(tiles)]) if len(tiles) else np.zeros((0, 4, 50, 3), np.float32)
    tiles = np.sort(tiles)
    X = np.concatenate([make_cnn8(I[:, k]) for k in range(4)], 0)
    tile = np.concatenate([tiles] * 4); sens = np.repeat(EMB_OF_K, len(tiles))
    yy = np.concatenate([y_uec[tiles]] * 4)
    return X, tile, sens, yy

Xtr, tile_tr, sens_tr, y_tr = build_samples(tr_tiles)
cnn_mean = Xtr.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
cnn_std = np.maximum(Xtr.std(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None], 1e-6)
sc_by = SC_raw[tile_tr]
sc_mean = sc_by.mean(0, dtype=np.float64).astype(np.float32); sc_std = np.maximum(sc_by.std(0, dtype=np.float64).astype(np.float32), 1e-6)
SC = torch.from_numpy(((SC_raw - sc_mean) / sc_std).astype(np.float32))
np.savez(os.path.join(OUT, "feature_normalization.npz"), cnn_mean=cnn_mean, cnn_std=cnn_std, scalar_mean=sc_mean, scalar_std=sc_std)
def to_t(X): return torch.from_numpy(((X - cnn_mean) / cnn_std).astype(np.float32))
Xtr_t = to_t(Xtr); del Xtr
tile_tr_t = torch.from_numpy(tile_tr); sens_tr_t = torch.from_numpy(sens_tr); y_tr_t = torch.from_numpy(y_tr)
if len(va_tiles):
    Xva, tile_va, sens_va, y_va = build_samples(va_tiles)
    Xva_t = to_t(Xva); del Xva
    tile_va_t = torch.from_numpy(tile_va); sens_va_t = torch.from_numpy(sens_va)
log(f"samples train={len(y_tr)} val={len(va_tiles)*4} train_classes={np.bincount(y_tr, minlength=NC).tolist()}")

# ---------------- model (verbatim structure from the repo)
class ConvBlock(nn.Module):
    def __init__(self, i, o, k, dropout):
        super().__init__(); p = k // 2
        self.net = nn.Sequential(nn.Conv1d(i, o, k, padding=p, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Conv1d(o, o, k, padding=p, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

def make_video_projection(a):
    vh = a.video_hidden_dim
    return nn.ModuleDict({
        "mean": nn.Sequential(nn.Linear(VIDEO_DIM, vh), nn.LayerNorm(vh), nn.ReLU(), nn.Dropout(a.video_dropout)),
        "std": nn.Sequential(nn.Linear(VIDEO_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(a.video_dropout)),
        "delta5": nn.Sequential(nn.Linear(VIDEO_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(a.video_dropout)),
        "scalar": nn.Sequential(nn.Linear(VIDEO_SCALAR_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(a.video_dropout)),
        "out": nn.Sequential(nn.Linear(vh + vh // 2 + vh // 2 + vh // 2, a.video_out_dim), nn.LayerNorm(a.video_out_dim), nn.ReLU(), nn.Dropout(a.video_dropout)),
    })

class CNN8VideoAuxNet(nn.Module):
    def __init__(self, a):
        super().__init__(); base = a.cnn_base_channels
        self.cnn = nn.Sequential(ConvBlock(8, base, 5, a.cnn_dropout), nn.MaxPool1d(2),
                                 ConvBlock(base, base * 2, 3, a.cnn_dropout), nn.MaxPool1d(2),
                                 ConvBlock(base * 2, base * 2, 3, a.cnn_dropout))
        self.sensor_emb = nn.Embedding(4, a.sensor_emb_dim)
        total = base * 2 * 2 + a.video_out_dim + a.sensor_emb_dim
        self.video_projection = make_video_projection(a)
        h = a.classifier_hidden
        self.classifier = nn.Sequential(nn.Linear(total, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(a.classifier_dropout),
                                        nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(a.classifier_dropout),
                                        nn.Linear(h // 2, NC))
    def forward(self, x, vm, vs, vd, sc, sens):
        h = self.cnn(x)
        inert = torch.cat([F.adaptive_avg_pool1d(h, 1).squeeze(-1), F.adaptive_max_pool1d(h, 1).squeeze(-1)], 1)
        p = self.video_projection
        v = p["out"](torch.cat([p["mean"](vm), p["std"](vs), p["delta5"](vd), p["scalar"](sc)], 1))
        return self.classifier(torch.cat([inert, v, self.sensor_emb(sens)], 1))

model = CNN8VideoAuxNet(args)
counts = np.maximum(np.bincount(y_tr, minlength=NC).astype(np.float32), 1.0)
cw = len(y_tr) / (NC * counts); cw = np.sqrt(cw); cw = cw / cw.mean()     # class_weight sqrt
crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32), label_smoothing=args.label_smoothing)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience)
replay = None
if args.lr_replay:
    h = pd.read_csv(args.lr_replay)
    replay = [args.learning_rate] + h["lr"].tolist()   # lr used in epoch e = lr after scheduler step of epoch e-1

@torch.inference_mode()
def predict(X, tile, sens, bs=4096):
    model.eval(); out = []
    for b in range(0, len(X), bs):
        tt = tile[b:b + bs]
        out.append(torch.softmax(model(X[b:b + bs], VM[tt], VS[tt], VD[tt], SC[tt], sens[b:b + bs]), 1))
    return torch.cat(out).numpy()

best, best_ep, bad, hist, best_state = -1.0, None, 0, [], None
n = len(y_tr); g = torch.Generator().manual_seed(args.seed)
for ep in range(1, args.epochs + 1):
    if replay is not None:
        lr_now = replay[min(ep - 1, len(replay) - 1)]
        for pg in opt.param_groups: pg["lr"] = lr_now
    lr_used = opt.param_groups[0]["lr"]
    model.train(); tl = 0.0; nb = 0; te = time.time()
    order = torch.randperm(n, generator=g)
    for b in range(0, n, args.batch_size):
        idx = order[b:b + args.batch_size]
        if len(idx) < 2: continue
        tt = tile_tr_t[idx]
        logits = model(Xtr_t[idx], VM[tt], VS[tt], VD[tt], SC[tt], sens_tr_t[idx])
        loss = crit(logits, y_tr_t[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl += float(loss.item()); nb += 1
    row = {"epoch": ep, "train_loss": tl / max(nb, 1), "lr_used": lr_used, "sec": time.time() - te}
    if len(va_tiles):
        P = predict(Xva_t, tile_va_t, sens_va_t)
        f = f1_score(y_va, P.argmax(1), average="macro", zero_division=0)
        acc = accuracy_score(y_va, P.argmax(1))
        sched.step(f)
        row.update({"val_macro_f1": f, "val_acc": acc})
        if f > best:
            best, best_ep, bad = f, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "best_epoch": ep, "best_macro_f1": f}, os.path.join(OUT, "best.pt"))
            np.save(os.path.join(OUT, "val_probabilities.npy"), P.astype(np.float32))
        else:
            bad += 1
    row["lr"] = opt.param_groups[0]["lr"]
    hist.append(row); pd.DataFrame(hist).to_csv(os.path.join(OUT, "history.csv"), index=False)
    log(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if len(va_tiles) and bad >= args.early_stopping_rounds:
        log("early stopping"); break

if best_state is not None:
    model.load_state_dict(best_state)
    log(f"best epoch={best_ep} val_macro_f1={best:.6f}")
torch.save({"model_state_dict": model.state_dict()}, os.path.join(OUT, "final.pt"))

# ---------------- test (row i = test id i)
tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
assert (tm.id.to_numpy() == np.arange(len(tm))).all()
TI = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float32)
KEY = {"right_arm": 0, "left_arm": 1, "right_leg": 2, "left_leg": 3}
tsens = torch.from_numpy(tm.sensor_location.map(KEY).to_numpy().astype(np.int64))
Xte = to_t(make_cnn8(TI))
tVM = torch.from_numpy(np.load(os.path.join(CACHE, "test_vi_mean.npy"))); tVS = torch.from_numpy(np.load(os.path.join(CACHE, "test_vi_std.npy")))
tVD = torch.from_numpy(np.load(os.path.join(CACHE, "test_vi_d5.npy")))
tSC = torch.from_numpy(((np.load(os.path.join(CACHE, "test_scalar.npy")) - sc_mean) / sc_std).astype(np.float32))
model.eval(); outs = []
with torch.inference_mode():
    for b in range(0, len(tm), 4096):
        s = slice(b, b + 4096)
        outs.append(torch.softmax(model(Xte[s], tVM[s], tVS[s], tVD[s], tSC[s], tsens[s]), 1))
Pte = torch.cat(outs).numpy().astype(np.float32)
np.save(os.path.join(OUT, "test_probabilities.npy"), Pte)
log("test saved", Pte.shape, "pred dist", np.bincount(Pte.argmax(1), minlength=NC).tolist())

# ---------------- std-protocol OOF for the held-out fold (all 4 limbs, NaN where the limb has no IMU)
if args.mode == "std":
    rows = np.where(std_fold == args.fold)[0]
    I = np.asarray(imu[rows])
    oof = np.full((len(rows), 4, NC), np.nan, np.float32)
    with torch.inference_mode():
        for li, k in enumerate(LIMB_TO_K):
            valid = np.isfinite(I[:, k]).all(axis=(1, 2))
            if not valid.any(): continue
            X = to_t(make_cnn8(I[valid][:, k])); tt = torch.from_numpy(rows[valid]); ss = torch.full((len(tt),), int(EMB_OF_K[k]), dtype=torch.long)
            oof[valid, li] = predict(X, tt, ss)
    np.save(os.path.join(OUT, "oof_rows.npy"), rows); np.save(os.path.join(OUT, "oof_fold.npy"), oof)
    log("oof saved", oof.shape)
log("DONE")
