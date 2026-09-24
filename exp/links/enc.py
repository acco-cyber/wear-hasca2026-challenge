"""Contrastive bi-encoder for successor prediction: f(a) . g(b) high when b is the next second after a.
Full-session softmax (InfoNCE over every window of the session, both directions).
python enc.py --train sbj_x,... --out enc_A.pt [--steps 800]
"""
import argparse, time
from lk import *
import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(3)

FRAMES = [0, 1, 2, 4, 7, 10, 12, 13, 14]
DROP = [0.2, 0.1]   # hidden dropout, input dropout (inference unaffected)
WD = [1e-4]

def session_inputs(V, X4=None, Xi=None, limb=None):
    """V (n,15,160) video PCA; either X4 (n,4,50,3) all limbs (training) or Xi (n,50,3)+limb (eval/test).
    Returns video part (n, 2400+160) and imu part (n,4,154) or (n,154)."""
    V = np.asarray(V, np.float32); V = V - V.reshape(-1, 160).mean(0)[None, None]   # per-session centring
    V = V / 1.5
    vpart = np.concatenate([V[:, FRAMES].reshape(len(V), -1), (V[:, -1] - V[:, 0]), V.mean(1)], 1)
    def imu_feat(X, lb):
        X = np.nan_to_num(np.asarray(X, np.float32)) / 10.0
        oh = np.zeros((len(X), 4), np.float32); oh[np.arange(len(X)), lb] = 1
        return np.concatenate([X.reshape(len(X), -1), oh], 1)
    if X4 is not None:
        return vpart, np.stack([imu_feat(X4[:, l], np.full(len(X4), l)) for l in range(4)], 1)
    return vpart, imu_feat(Xi, limb)

class Enc(nn.Module):
    def __init__(self, din, h=256, d=96, p=None, pin=None):
        super().__init__()
        p = DROP[0] if p is None else p; pin = DROP[1] if pin is None else pin
        self.net = nn.Sequential(nn.Dropout(pin), nn.Linear(din, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))
    def forward(self, x): return Fn.normalize(self.net(x), dim=-1)

class BiEnc(nn.Module):
    def __init__(self, din):
        super().__init__(); self.f = Enc(din); self.g = Enc(din); self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))
    def sims(self, x):
        return self.f(x), self.g(x)

def score_matrix(model, vpart, ipart):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.concatenate([vpart, ipart], 1)); F, G = model.sims(x)
        S = (F @ G.T).numpy() * float(model.logit_scale.exp())
    return S

def valid_limb_choice(X4, rng):
    n = len(X4); ok = ~np.isnan(X4).any(axis=(2, 3))   # (n,4)
    r = rng.rand(n, 4) * ok; r[~ok.any(1)] = rng.rand((~ok.any(1)).sum(), 4)
    return r.argmax(1)

def train(sessions, steps=800, seed=0, lr=1e-3, log_every=100, batch=700):
    meta, imu, vid = load_prep(); sl = session_slices(meta); rng = np.random.RandomState(seed); torch.manual_seed(seed)
    data = {}
    for s in sessions:
        a, b = sl[s]; vp, ip = session_inputs(vid[a:b], X4=np.asarray(imu[a:b], np.float32))
        data[s] = (torch.from_numpy(vp), torch.from_numpy(ip), np.asarray(imu[a:b], np.float32))
    din = data[sessions[0]][0].shape[1] + data[sessions[0]][1].shape[2]
    model = BiEnc(din); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD[0])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    sizes = np.array([len(data[s][0]) for s in sessions], float); t0 = time.time(); run = []
    for step in range(steps):
        model.train(); s = sessions[rng.choice(len(sessions), p=sizes / sizes.sum())]
        vp, ip, X4 = data[s]; n = len(vp)
        anc = np.sort(rng.choice(n - 1, min(n - 1, batch), replace=False))
        pool = np.unique(np.concatenate([anc, anc + 1])); pos = {int(v): k for k, v in enumerate(pool)}
        ia = torch.tensor([pos[int(i)] for i in anc]); ib = torch.tensor([pos[int(i) + 1] for i in anc])
        lb = valid_limb_choice(X4[pool], rng)
        tp = torch.from_numpy(pool)
        x = torch.cat([vp[tp], ip[tp, torch.from_numpy(lb)]], 1); m = len(pool)
        F, G = model.sims(x); sc = model.logit_scale.exp().clamp(max=100)
        L = (F @ G.T) * sc
        L = L - torch.eye(m) * 1e4
        loss = 0.5 * (Fn.cross_entropy(L[ia], ib) + Fn.cross_entropy(L.T[ib], ia))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run.append(float(loss.detach()))
        if (step + 1) % log_every == 0 or step == 19:
            print(f"  step {step+1} loss {np.mean(run[-log_every:]):.3f} scale {float(sc):.1f} ({time.time()-t0:.0f}s)", flush=True)
    return model

def rank_stats(S):
    n = len(S); S = S.copy(); np.fill_diagonal(S, -np.inf); tr = S[np.arange(n - 1), np.arange(1, n)]
    r = (S[:-1] > tr[:, None]).sum(1)
    return dict(top1=(r < 1).mean(), top10=(r < 10).mean(), top40=(r < 40).mean(), top100=(r < 100).mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=800); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drop", default="0.2,0.1"); ap.add_argument("--wd", type=float, default=1e-4)
    a = ap.parse_args(); DROP[:] = [float(x) for x in a.drop.split(",")]; WD[0] = a.wd
    tr = a.train.split(","); model = train(tr, steps=a.steps, seed=a.seed, lr=a.lr)
    torch.save(model.state_dict(), os.path.join(EXP, a.out))
    # quick eval on the eval sessions with the stored sim limbs
    S = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); meta, imu, vid = load_prep(); rows = []
    for s in EVAL:
        st = S[s]; a_, b_ = st["a"], st["b"]; X4 = np.asarray(imu[a_:b_], np.float32)
        vp, ip = session_inputs(vid[a_:b_], Xi=X4[np.arange(st["n"]), st["limb"]], limb=st["limb"])
        rows.append(dict(session=s, **rank_stats(score_matrix(model, vp, ip))))
    df = pd.DataFrame(rows); print(df.round(3).to_string()); print("mean", df.mean(numeric_only=True).round(3).to_dict())
