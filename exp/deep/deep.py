"""CPU deep window experts (public 'hungarian-chain' recipe, shrunk for CPU).
python deep.py <expert: vid|imu> <mode: f0|full> [--epochs E] [--seed S] [--threads T] ...
f0   : train on folds 1-4, monitor/evaluate on fold 0 (subjects 9,15,16,20,21) every epoch, save fold-0 preds.
full : train on every subject for E epochs, write test probs (12234,19) in test-id order.
Outputs -> exp/deep/<expert>_<mode>_s<seed>[_tag]/"""
import os, sys, time, argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("expert"); ap.add_argument("mode")
ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--tag", default="")
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--null_ratio", type=float, default=2.5)
ap.add_argument("--wpow", type=float, default=0.5); ap.add_argument("--ls", type=float, default=0.05)
ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--hid", type=int, default=96); ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--limbs_per_sec", type=int, default=2); ap.add_argument("--noise", type=float, default=0.1)
ap.add_argument("--shift", type=int, default=1); ap.add_argument("--subj_center", type=int, default=0)
a = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(a.threads); os.environ["MKL_NUM_THREADS"] = str(a.threads)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
torch.set_num_threads(a.threads)

PREP = r"E:\Claude code\wear\data\prep"; TEST = r"E:\Claude code\wear\data\test"; WORK = r"E:\Claude code\wear\work"
EXPD = r"E:\Claude code\wear\exp\deep"
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]; NC = 19
name = f"{a.expert}_{a.mode}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
OD = os.path.join(EXPD, name); os.makedirs(OD, exist_ok=True)
t0 = time.time()
def log(*s):
    print(f"[{time.time()-t0:6.0f}s]", *s, flush=True)

m = pd.read_csv(os.path.join(PREP, "train_meta.csv"))
N = len(m); y = m.y.to_numpy(); pur = m.pur.to_numpy(); sbj = m.sbj.to_numpy(); ses = m.session.to_numpy(); tt = m.t.to_numpy()
perm = np.random.RandomState(0).permutation(np.unique(sbj)); fo = {s: i % 5 for i, s in enumerate(perm)}
fold = np.array([fo[s] for s in sbj])
# fixed single-limb pick identical to exp/base/common.eval_single (seed 5 on lgbm_v1 NaN pattern)
ref = np.load(os.path.join(WORK, "lgbm_v1", "oof.npy"), mmap_mode="r")
refv = ~np.isnan(np.asarray(ref[:, :, 0])); rng5 = np.random.RandomState(5)
RL = np.array([rng5.choice(np.where(v)[0]) if v.any() else -1 for v in refv]); del ref
HOLD = a.mode != "full"; K = int(a.mode[1:]) if HOLD else -1  # mode fK = hold out fold K
EVAL = np.where((RL >= 0) & (pur >= 0.8) & (fold == K))[0]

tr_sec = np.where((fold != K) if HOLD else np.ones(N, bool))[0]
tr_sec = tr_sec[pur[tr_sec] >= 0.8]
va_sec = np.where(fold == K)[0] if HOLD else np.zeros(0, int)
rs = np.random.RandomState(a.seed * 7919 + 13); torch.manual_seed(a.seed * 7919 + 13)

def capped(secs):
    """Per subject: keep all activity seconds, a fresh random subset of null <= ratio * largest class."""
    out = []
    for s in np.unique(sbj[secs]):
        g = secs[sbj[secs] == s]; nul = g[y[g] == 0]; act = g[y[g] != 0]
        cap = int(round(a.null_ratio * np.bincount(y[act], minlength=NC)[1:].max())) if len(act) else len(nul)
        if len(nul) > cap: nul = rs.choice(nul, cap, replace=False)
        out.append(act); out.append(nul)
    return np.concatenate(out)

cnt = np.zeros(NC)
for _ in range(3): cnt += np.bincount(y[capped(tr_sec)], minlength=NC)
cw = (np.maximum(cnt, 1) ** -a.wpow); cw = cw / (cw * cnt).sum() * cnt.sum()
CW = torch.tensor(cw, dtype=torch.float32)
log(name, "train secs", len(tr_sec), "val secs", len(va_sec), "class w", np.round(cw, 2).tolist())

# ------------------------------------------------------------------ models
class VidNet(nn.Module):
    def __init__(s, din=160, d=128, L=2, drop=0.1):
        super().__init__()
        s.proj = nn.Linear(din, d); s.pos = nn.Parameter(torch.zeros(1, 15, d)); nn.init.trunc_normal_(s.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(d); s.score = nn.Linear(d, 1)
        s.stat = nn.Sequential(nn.LayerNorm(din * 3), nn.Linear(din * 3, d), nn.GELU(), nn.Dropout(drop))
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * d, NC))
    def forward(s, x):
        h = s.norm(s.enc(s.proj(x) + s.pos)); w = torch.softmax(s.score(h).squeeze(-1), 1)
        tv = (h * w.unsqueeze(-1)).sum(1)
        st = torch.cat([x.mean(1), x.std(1), (x[:, 1:] - x[:, :-1]).abs().mean(1)], 1)
        return s.head(torch.cat([tv, s.stat(st)], 1))

class ImuNet(nn.Module):
    def __init__(s, d=128, h=96, L=2, drop=0.1):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(4, 64, 5, padding=2), nn.GELU(), nn.Conv1d(64, d, 5, stride=5))
        s.pos = nn.Parameter(torch.zeros(1, 10, d)); nn.init.trunc_normal_(s.pos, std=0.02)
        s.limb = nn.Embedding(4, d); s.limb2 = nn.Embedding(4, 32)
        s.lstm = nn.LSTM(d, h, L, batch_first=True, bidirectional=True, dropout=drop)
        s.score = nn.Linear(2 * h, 1)
        s.head = nn.Sequential(nn.Linear(4 * h + 32, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, NC))
    def forward(s, x, l):
        z = torch.cat([x, x.norm(dim=-1, keepdim=True)], -1).transpose(1, 2)
        tok = s.stem(z).transpose(1, 2) + s.pos + s.limb(l).unsqueeze(1)
        h, _ = s.lstm(tok); w = torch.softmax(s.score(h).squeeze(-1), 1)
        return s.head(torch.cat([(h * w.unsqueeze(-1)).sum(1), h.mean(1), s.limb2(l)], 1))

# ------------------------------------------------------------------ data
te_meta = pd.read_csv(os.path.join(TEST, "test_meta_data.csv"))
assert (te_meta.id.to_numpy() == np.arange(len(te_meta))).all()
te_limb = np.array([LIMBS.index(v) for v in te_meta.sensor_location])
te_sbj = te_meta.sbj_id.to_numpy()

if a.expert == "vid":
    V = np.load(os.path.join(PREP, "train_vid_pca.npy")).astype(np.float32)
    Vt = np.load(os.path.join(PREP, "test_vid_pca.npy")).astype(np.float32)
    if a.subj_center:  # remove each subject's mean frame (train: per subject; test: per test subject)
        for s in np.unique(sbj): V[sbj == s] -= np.median(V[sbj == s].mean(1), 0)[None, None, :]
        for s in np.unique(te_sbj): Vt[te_sbj == s] -= np.median(Vt[te_sbj == s].mean(1), 0)[None, None, :]
    mu = V[tr_sec].mean((0, 1)); sd = V[tr_sec].std((0, 1)) + 1e-3
    V = np.clip((V - mu) / sd, -8, 8); Vt = np.clip((Vt - mu) / sd, -8, 8)
    V = torch.from_numpy(V); Vt = torch.from_numpy(Vt)
    net = VidNet(160, a.d, a.layers, a.drop)
else:
    X = np.load(os.path.join(PREP, "train_imu.npy")).astype(np.float32)  # (N,4,50,3)
    valid = ~np.isnan(X).any(axis=(2, 3)); X = np.nan_to_num(X)
    Xt = np.load(os.path.join(TEST, "test_inertial_data.npy")).astype(np.float32)
    nxt = np.full(N, -1); same = (ses[1:] == ses[:-1]) & (tt[1:] == tt[:-1] + 1); nxt[:-1][same] = np.arange(1, N)[same]
    # time-shift allowed where second and its successor are both pure with the same label and the limb is valid
    shiftok = np.zeros((N, 4), bool)
    j = np.where(nxt >= 0)[0]
    okj = (y[nxt[j]] == y[j]) & (pur[j] >= 0.999) & (pur[nxt[j]] >= 0.999)
    shiftok[j] = okj[:, None] & valid[j] & valid[nxt[j]]
    X2 = np.concatenate([X, X[np.maximum(nxt, 0)]], axis=2)  # (N,4,100,3)
    X = torch.from_numpy(X); X2t = torch.from_numpy(X2); Xt = torch.from_numpy(Xt)
    net = ImuNet(a.d, a.hid, a.layers, a.drop)
log("params", sum(p.numel() for p in net.parameters()))

def epoch_rows():
    secs = capped(tr_sec)
    if a.expert == "vid": return secs[rs.permutation(len(secs))], None
    r = rs.rand(len(secs), 4); r[~valid[secs]] = 9
    order = np.argsort(r, 1)[:, :a.limbs_per_sec]
    S = np.repeat(secs, order.shape[1]); L = order.ravel()
    ok = valid[S, L]; S, L = S[ok], L[ok]; p = rs.permutation(len(S))
    return S[p], L[p]

def batch_x(S, L, train):
    if a.expert == "vid":
        x = V[S]
        if train and a.noise > 0: x = x + a.noise * torch.randn_like(x)
        return (x,)
    if train and a.shift:
        off = np.where(shiftok[S, L], rs.randint(0, 50, len(S)), 0)
        idx = torch.from_numpy(off[:, None] + np.arange(50)[None, :])
        w = X2t[torch.from_numpy(S), torch.from_numpy(L)]  # (B,100,3)
        x = torch.gather(w, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    else:
        x = X[torch.from_numpy(S), torch.from_numpy(L)]
    if train:
        x = x * torch.from_numpy(rs.uniform(0.9, 1.1, (len(S), 1, 1)).astype(np.float32))
        if a.noise > 0: x = x + 0.02 * torch.randn_like(x)
    return x, torch.from_numpy(L).long()

@torch.no_grad()
def predict_train(secs):
    """(len(secs),4,19) probs; vid broadcast over valid limbs; NaN for invalid limbs (imu) ."""
    net.eval(); out = np.full((len(secs), 4, NC), np.nan, np.float32)
    if a.expert == "vid":
        P = np.concatenate([torch.softmax(net(V[secs[i:i + 2048]]), 1).numpy() for i in range(0, len(secs), 2048)])
        out[:] = P[:, None, :]
        return out
    for l in range(4):
        ss = secs[valid[secs, l]]; li = np.full(len(ss), l)
        P = np.concatenate([torch.softmax(net(*batch_x(ss[i:i + 4096], li[i:i + 4096], False)), 1).numpy()
                            for i in range(0, len(ss), 4096)]) if len(ss) else np.zeros((0, NC))
        out[np.searchsorted(secs, ss), l] = P
    return out

@torch.no_grad()
def predict_test():
    net.eval()
    if a.expert == "vid":
        return np.concatenate([torch.softmax(net(Vt[i:i + 2048]), 1).numpy() for i in range(0, len(Vt), 2048)])
    return np.concatenate([torch.softmax(net(Xt[i:i + 4096], torch.from_numpy(te_limb[i:i + 4096]).long()), 1).numpy()
                           for i in range(0, len(Xt), 4096)])

def f0_score(P):  # P (len(va_sec),4,19) aligned with va_sec
    pos = np.searchsorted(va_sec, EVAL); Q = P[pos, RL[EVAL]]
    Q = np.where(np.isnan(Q[:, :1]), 1.0 / NC, Q)
    return f1_score(y[EVAL], Q.argmax(1), average="macro"), f1_score(y[EVAL], Q.argmax(1), average=None, labels=range(NC))

# ------------------------------------------------------------------ train
S0, _ = epoch_rows(); steps_per = (len(S0) * (1 if a.expert == "vid" else 1) + a.bs - 1) // a.bs
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per + 50, pct_start=0.1)
hist = []; best = (-1, None, None)
for ep in range(a.epochs):
    net.train(); S, L = epoch_rows(); tl = 0.0; nb = 0
    for i in range(0, len(S), a.bs):
        sb = S[i:i + a.bs]; lb = None if L is None else L[i:i + a.bs]
        xb = batch_x(sb, lb, True); yb = torch.from_numpy(y[sb])
        loss = F.cross_entropy(net(*xb), yb, weight=CW, label_smoothing=a.ls)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if nb < a.epochs * steps_per + 49: sch.step()
        tl += loss.item(); nb += 1
    msg = f"ep {ep+1}/{a.epochs} rows {len(S)} loss {tl/nb:.4f}"
    if HOLD:
        P = predict_train(va_sec); f, pc = f0_score(P); hist.append(f); msg += f" fold{K} single-limb F1 {f:.4f}"
        if f > best[0]: best = (f, ep + 1, P)
        if K == 0: np.save(os.path.join(OD, f"f0_ep{ep+1}.npy"), P.astype(np.float16))
    log(msg)

torch.save(net.state_dict(), os.path.join(OD, "model.pt"))
res = {"args": vars(a), "hist": hist}
if HOLD:
    P = predict_train(va_sec); f, pc = f0_score(P)
    full = np.full((N, 4, NC), np.nan, np.float32); full[va_sec] = P
    np.save(os.path.join(OD, f"oof_f{K}.npy"), full)
    res.update(final=f, best=best[0], best_ep=best[1], per_class=np.round(pc, 3).tolist())
    log(f"FINAL fold{K} single-limb F1", round(f, 4), "best", round(best[0], 4), "@ep", best[1])
    log("per-class", np.round(pc, 2).tolist())
T = predict_test(); np.save(os.path.join(OD, "test.npy"), T.astype(np.float32))
log("test pred dist", np.bincount(T.argmax(1), minlength=NC).tolist())
json.dump(res, open(os.path.join(OD, "res.json"), "w"), indent=1)
log("done")
