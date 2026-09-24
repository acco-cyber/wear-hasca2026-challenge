"""Faithful local rebuild of nomannic19 'ts-emb-3wdc-temporal-fusion-ensemble' (LB 0.61278).
Pooled MLP (3 ep) + Inception1D/Transformer temporal fusion (4 ep), 4 limbs as 4 samples + sensor embedding,
blend 0.6 temporal / 0.4 pooled, probs[:,0] *= exp(0.75).
Local adaptations: stride-25 windows rebuilt from 1-s tiles (offset window = 2nd half of t + 1st half of t+1,
video = frames 16-22 of t + 38-45 of t+1), video = PCA-160 reconstructed to 768 (same for train and test).
usage: python nom_train.py --seed 0 --threads 4 --out nom_full/s0 [--wps 150] [--holdout 2,9,17,21]
"""
import argparse, os, time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--out', required=True)
ap.add_argument('--wps', type=int, default=150)  # windows per (subject,class); 0 = all
ap.add_argument('--holdout', default='')
ap.add_argument('--pe', type=int, default=3)
ap.add_argument('--te', type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
SEED = 42 + args.seed
np.random.seed(SEED); torch.manual_seed(SEED)
OUT = os.path.join(r"E:\Claude code\wear\exp\nom", args.out); os.makedirs(OUT, exist_ok=True)
D = r"E:\Claude code\wear\data"
N_CLASSES = 19
NB_ORDER = [2, 3, 1, 0]  # ours [left_arm,left_leg,right_arm,right_leg] -> nb [right_arm,right_leg,left_leg,left_arm]
SENSOR_TO_ID = {"right_arm": 0, "right_leg": 1, "left_leg": 2, "left_arm": 3}
t0 = time.time()
def log(*a):
    s = f"[{time.time()-t0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(OUT, 'log.txt'), 'a') as f: f.write(s + "\n")

meta = pd.read_csv(D + r"\prep\train_meta.csv")
imu = np.load(D + r"\prep\train_imu.npy")            # (N,4,50,3) f16
vp = np.load(D + r"\prep\train_vid_pca.npy")          # (N,15,160) f16
comp = torch.tensor(np.load(D + r"\prep\pca_components.npy"), dtype=torch.float32)
mu = torch.tensor(np.load(D + r"\prep\pca_mean.npy"), dtype=torch.float32)
def recon(v):  # (B,15,160) -> (B,15,768)
    return v @ comp + mu

sess = meta.session.values; sbj = meta.sbj.values; y = meta.y.values; pur = meta.pur.values; tt = meta.t.values
N = len(meta)
hold = [int(s) for s in args.holdout.split(',')] if args.holdout else []
trmask = ~np.isin(sbj, hold)
# candidates: aligned (off=0) and offset (off=1) windows fully inside one label segment
al = np.flatnonzero((pur >= 1.0) & trmask)
nxt = np.r_[sess[1:] == sess[:-1], False] & np.r_[tt[1:] == tt[:-1] + 1, False]
yn = np.r_[y[1:], -1]; pn = np.r_[pur[1:], 0]
of = np.flatnonzero(nxt & (pur >= 1.0) & (pn >= 1.0) & (y == yn) & trmask)
cand = pd.DataFrame({'i': np.r_[al, of], 'off': np.r_[np.zeros(len(al), int), np.ones(len(of), int)]})
cand['sbj'] = sbj[cand.i.values]; cand['y'] = y[cand.i.values]
rng = np.random.RandomState(SEED)
parts = []
for (s, c), g in cand.groupby(['sbj', 'y']):
    if args.wps > 0 and len(g) > args.wps:
        g = g.iloc[rng.choice(len(g), args.wps, replace=False)]
    parts.append(g)
tr = pd.concat(parts, ignore_index=True)
tr = tr.iloc[rng.permutation(len(tr))].reset_index(drop=True)
log(f"candidates aligned {len(al)} offset {len(of)} | train windows {len(tr)} | subjects {tr.sbj.nunique()} holdout {hold}")

def build(ii, off):
    X = np.empty((len(ii), 4, 50, 3), np.float32); V = np.empty((len(ii), 15, 160), np.float32)
    m0 = off == 0; m1 = ~m0
    X[m0] = imu[ii[m0]]; V[m0] = vp[ii[m0]]
    a = ii[m1]
    X[m1] = np.concatenate([imu[a][:, :, 25:], imu[a + 1][:, :, :25]], axis=2)
    V[m1] = np.concatenate([vp[a][:, 8:15], vp[a + 1][:, 0:8]], axis=1)
    X = X[:, NB_ORDER].transpose(0, 1, 3, 2)  # (M,4,3,50)
    valid = np.isfinite(X).all(axis=(2, 3))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(np.ascontiguousarray(X)), torch.tensor(V), torch.tensor(valid)
Xtr, Vtr, VALtr = build(tr.i.values, tr.off.values)
Ytr = torch.tensor(tr.y.values, dtype=torch.long)
log('built', Xtr.shape, Vtr.shape, 'valid frac', VALtr.float().mean().item())

# ---------------- models (verbatim from notebook) ----------------
class PooledFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inertial_encoder = nn.Sequential(nn.Linear(16, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.15))
        self.video_encoder = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 192), nn.GELU(), nn.Dropout(.2))
        self.sensor_embedding = nn.Embedding(4, 8)
        self.classifier = nn.Sequential(nn.Linear(264, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, N_CLASSES))
    def forward(self, inertial, video, sensor_ids=None):
        batch_size, sensors = inertial.shape[:2]
        magnitude = torch.linalg.vector_norm(inertial, dim=2)
        inertial_features = torch.cat([
            inertial.mean(-1), inertial.std(-1, unbiased=False), inertial.amin(-1), inertial.amax(-1),
            magnitude.mean(-1, keepdim=True), magnitude.std(-1, unbiased=False, keepdim=True),
            magnitude.amin(-1, keepdim=True), magnitude.amax(-1, keepdim=True)], dim=-1)
        inertial_features = self.inertial_encoder(inertial_features)
        video_features = self.video_encoder(torch.cat([video.mean(1), video.std(1, unbiased=False)], dim=-1))
        video_features = video_features[:, None].expand(-1, sensors, -1)
        if sensor_ids is None:
            sensor_ids = torch.arange(sensors, device=inertial.device)[None].expand(batch_size, -1)
        fused = torch.cat([inertial_features, video_features, self.sensor_embedding(sensor_ids)], dim=-1)
        return self.classifier(fused).reshape(-1, N_CLASSES)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, branch_channels=32):
        super().__init__()
        out_channels = branch_channels * 4
        self.bottleneck = nn.Conv1d(in_channels, branch_channels, 1, bias=False)
        self.branches = nn.ModuleList([nn.Conv1d(branch_channels, branch_channels, k, padding=k // 2, bias=False) for k in (5, 11, 21)])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, branch_channels, 1, bias=False))
        self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.norm = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(.1)
    def forward(self, x):
        reduced = self.bottleneck(x)
        merged = torch.cat([b(reduced) for b in self.branches] + [self.pool_branch(x)], dim=1)
        return self.dropout(F.gelu(self.norm(merged + self.skip(x))))

class InertialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(InceptionBlock(4), InceptionBlock(128))
        self.head = nn.Sequential(nn.Linear(256, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.2))
    def forward(self, x):
        magnitude = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.blocks(torch.cat([x, magnitude], dim=1))
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], dim=1))

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 192), nn.GELU())
        self.position = nn.Parameter(torch.zeros(1, 15, 192))
        layer = nn.TransformerEncoderLayer(192, 4, 384, dropout=.15, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.attention = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.2))
        nn.init.trunc_normal_(self.position, std=.02)
    def forward(self, x):
        x = self.temporal(self.projection(x) + self.position)
        weights = torch.softmax(self.attention(x), dim=1)
        return self.head(torch.cat([(x * weights).sum(dim=1), x.mean(dim=1)], dim=1))

class TemporalFusionModel(nn.Module):
    def __init__(self, modality_dropout=.1):
        super().__init__()
        self.inertial_encoder, self.video_encoder = InertialEncoder(), VideoEncoder()
        self.sensor_embedding = nn.Embedding(4, 16)
        self.gate = nn.Linear(400, 192)
        self.classifier = nn.Sequential(nn.Linear(400, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.3), nn.Linear(192, N_CLASSES))
        self.modality_dropout = modality_dropout
    def forward(self, inertial, video, sensor_ids=None):
        batch_size, sensors = inertial.shape[:2]
        if self.training:
            scale = torch.empty(batch_size, sensors, 1, 1, device=inertial.device).uniform_(.9, 1.1)
            inertial = inertial * scale + torch.randn_like(inertial) * .01
        inertial_features = self.inertial_encoder(inertial.reshape(-1, 3, 50)).reshape(batch_size, sensors, -1)
        video_features = self.video_encoder(video)
        if self.training and self.modality_dropout:
            ik = (torch.rand(batch_size, sensors, 1) > self.modality_dropout) / (1 - self.modality_dropout)
            vk = (torch.rand(batch_size, 1) > self.modality_dropout) / (1 - self.modality_dropout)
            inertial_features = inertial_features * ik
            video_features = video_features * vk
        video_features = video_features[:, None].expand(-1, sensors, -1)
        if sensor_ids is None:
            sensor_ids = torch.arange(sensors, device=inertial.device)[None].expand(batch_size, -1)
        sf = self.sensor_embedding(sensor_ids)
        gate = torch.sigmoid(self.gate(torch.cat([inertial_features, video_features, sf], dim=-1)))
        fused = gate * inertial_features + (1 - gate) * video_features
        features = torch.cat([fused, inertial_features * video_features, sf], dim=-1)
        return self.classifier(features).reshape(-1, N_CLASSES)

def fit_model(model, epochs, lr, wd, name, cosine_tmax=None, bs=256, cb=None):
    crit = nn.CrossEntropyLoss(label_smoothing=.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_tmax, eta_min=lr * .1) if cosine_tmax else None
    n = len(Ytr)
    for ep in range(1, epochs + 1):
        model.train(); tl, ti = 0.0, 0
        perm = torch.randperm(n)
        for k in range(0, n, bs):
            b = perm[k:k + bs]
            x = Xtr[b]; v = recon(Vtr[b]); tgt = Ytr[b].repeat_interleave(4); val = VALtr[b].reshape(-1)
            opt.zero_grad(set_to_none=True)
            logits = model(x, v)
            loss = crit(logits[val], tgt[val])
            loss.backward(); opt.step()
            it = int(val.sum()); tl += loss.item() * it; ti += it
            if (k // bs) % 50 == 49: log(f"  {name} ep{ep} batch {k//bs+1}/{(n+bs-1)//bs} loss {tl/ti:.4f}")
        if sch: sch.step()
        log(f"{name} epoch {ep}/{epochs}: loss {tl/ti:.4f}")
        if cb: cb(model, ep)

@torch.no_grad()
def predict(model, X, V, S, bs=1024):  # X (M,1,3,50) V (M,15,160) S (M,1)
    model.eval(); out = []
    for k in range(0, len(X), bs):
        out.append(torch.softmax(model(X[k:k+bs], recon(V[k:k+bs]), S[k:k+bs]).float(), 1).numpy())
    return np.concatenate(out)

def combine(pp, pt):
    c = 0.6 * pt + 0.4 * pp
    cb = c.copy(); cb[:, 0] *= np.exp(0.75)
    return c, cb / cb.sum(1, keepdims=True)

if not hold:
    tm = pd.read_csv(D + r"\test\test_meta_data.csv")
    assert (tm.id.values == np.arange(len(tm))).all()
    Xt = np.load(D + r"\test\test_inertial_data.npy").astype(np.float32)
    Xt = torch.tensor(np.ascontiguousarray(np.nan_to_num(Xt, nan=0.0).transpose(0, 2, 1)[:, None]))
    St = torch.tensor(tm.sensor_location.map(SENSOR_TO_ID).values.astype(np.int64))[:, None]
    Vt = torch.tensor(np.load(D + r"\prep\test_vid_pca.npy").astype(np.float32))

pooled = PooledFusionModel()
fit_model(pooled, args.pe, 2e-3, 1e-4, 'pooled')
if not hold:
    pp_test = predict(pooled, Xt, Vt, St); np.save(os.path.join(OUT, 'test_pooled.npy'), pp_test); log('pooled test saved')
def ep_cb(model, ep):  # intermediate test probs (early deliverable)
    if hold or ep >= args.te or ep < 2: return
    pt_ = predict(model, Xt, Vt, St); c_, cb_ = combine(pp_test, pt_)
    np.save(os.path.join(OUT, f'test_ep{ep}.npy'), cb_); log(f'intermediate test_ep{ep} saved')
temporal = TemporalFusionModel()
fit_model(temporal, args.te, 8e-4, 1e-3, 'temporal', cosine_tmax=10, cb=ep_cb)
torch.save({'pooled': pooled.state_dict(), 'temporal': temporal.state_dict()}, os.path.join(OUT, 'models.pt'))

if hold:
    from sklearn.metrics import f1_score
    ii = np.flatnonzero(np.isin(sbj, hold))
    r2 = np.random.RandomState(123)
    limb = r2.randint(0, 4, len(ii))
    Xh = imu[ii, limb].astype(np.float32)
    bad = ~np.isfinite(Xh).all(axis=(1, 2))
    limb[bad] = 1; Xh = imu[ii, limb].astype(np.float32)
    Xh = torch.tensor(np.ascontiguousarray(Xh.transpose(0, 2, 1)[:, None]))
    inv = {0: 3, 1: 2, 2: 0, 3: 1}  # our limb -> nb sensor id
    Sh = torch.tensor(np.array([inv[l] for l in limb]))[:, None]
    Vh = torch.tensor(vp[ii].astype(np.float32))
    pp = predict(pooled, Xh, Vh, Sh); pt = predict(temporal, Xh, Vh, Sh)
    c, cb = combine(pp, pt); yh = y[ii]
    res = {k: float(f1_score(yh, p.argmax(1), average='macro')) for k, p in [('pooled', pp), ('temporal', pt), ('blend', c), ('blend_nullbias', cb)]}
    res['per_subject_blend_nb'] = {int(s): float(f1_score(yh[sbj[ii] == s], cb[sbj[ii] == s].argmax(1), average='macro')) for s in hold}
    log('HOLDOUT', json.dumps(res))
    np.save(os.path.join(OUT, 'hold_pp.npy'), pp); np.save(os.path.join(OUT, 'hold_pt.npy'), pt); np.save(os.path.join(OUT, 'hold_idx.npy'), ii)
else:
    pp = pp_test; pt = predict(temporal, Xt, Vt, St)
    c, cb = combine(pp, pt)
    np.save(os.path.join(OUT, 'test_pooled.npy'), pp); np.save(os.path.join(OUT, 'test_temporal.npy'), pt)
    np.save(os.path.join(OUT, 'test_nobias.npy'), c); np.save(os.path.join(OUT, 'test.npy'), cb)
    log('test saved', cb.shape, 'pred dist', np.bincount(cb.argmax(1), minlength=19).tolist())
log('done')
