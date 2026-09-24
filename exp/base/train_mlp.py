"""MLP base model on the v3 feature blocks (same 5 subject-disjoint folds). Rows = all valid (second, limb) pairs.
Class weights mimic the lgbm recipe (null capped at null_x * median activity count, sqrt-balanced) so the output prior
looks like lgbm_v1's (the decoder is tuned on that).
python train_mlp.py <tag> [--epochs 12] [--hid 512] [--drop 0.3] [--sim]"""
import os, sys, time, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch, torch.nn as nn, torch.nn.functional as Fnn
torch.set_num_threads(4)
from feats_v3 import train_blocks, test_blocks

class Net(nn.Module):
    def __init__(self, d, h, drop, k=NC):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop),
                               nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(h // 2, k))
    def forward(self, x): return self.f(x)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hid", type=int, default=512); ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--null_x", type=float, default=3.0); ap.add_argument("--ls", type=float, default=0.05)
    ap.add_argument("--in_drop", type=float, default=0.1); ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--sim", action="store_true"); ap.add_argument("--seed_offset", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    IM, VB = train_blocks(); IMt, VBt, tl = test_blocks()
    m = meta(); y = m.y.to_numpy(); pur = m.pur.to_numpy(); N = len(m); fold = sec_fold()
    ok = ~np.isnan(IM[:, :, 0]) & (pur >= 0.8)[:, None]
    def rows(s, l): return np.concatenate([IM[s, l], np.eye(4, dtype=np.float32)[l], VB[s]], 1)
    Xte = np.concatenate([IMt, np.eye(4, dtype=np.float32)[tl], VBt], 1)
    oof = np.full((N, 4, NC), np.nan, np.float32); test = np.zeros((len(Xte), NC), np.float32); cv = {}
    for k in range(5):
        trs, trl = np.where(ok & (fold != k)[:, None]); vas, val = np.where(ok & (fold == k)[:, None])
        Xtr = rows(trs, trl); Xva = rows(vas, val); ytr = y[trs]
        med = np.nanmedian(Xtr, 0); q1, q3 = np.nanpercentile(Xtr, [25, 75], axis=0); sc = (q3 - q1) + 1e-3
        prep = lambda X: torch.from_numpy(np.clip(np.nan_to_num((X - med) / sc), -6, 6).astype(np.float32))
        Ttr, Tva, Tte = prep(Xtr), prep(Xva), prep(Xte); del Xtr, Xva
        cnt = np.bincount(ytr, minlength=NC).astype(np.float64); eff = cnt.copy(); eff[0] = min(cnt[0], a.null_x * np.median(cnt[1:]))
        wc = np.sqrt(eff) / cnt; wc = wc / (wc[ytr].mean())
        W = torch.from_numpy(wc[ytr].astype(np.float32)); Y = torch.from_numpy(ytr)
        pva = np.zeros((len(Tva), NC), np.float32); pte = np.zeros((len(Tte), NC), np.float32)
        for sd in range(a.seeds):
            torch.manual_seed(sd * 100 + k + 1000 * a.seed_offset); net = Net(Ttr.shape[1], a.hid, a.drop)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
            bs = 512; steps = a.epochs * ((len(Ttr) + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
            for ep in range(a.epochs):
                net.train(); perm = torch.randperm(len(Ttr)); tl_ = 0.0
                for i in range(0, len(Ttr), bs):
                    b = perm[i:i + bs]; xb = Ttr[b]
                    if a.noise > 0: xb = xb + a.noise * torch.randn_like(xb)
                    if a.in_drop > 0: xb = xb * (torch.rand_like(xb) > a.in_drop).float() / (1 - a.in_drop)
                    loss = (Fnn.cross_entropy(net(xb), Y[b], reduction="none", label_smoothing=a.ls) * W[b]).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl_ += loss.item() * len(b)
                if ep == a.epochs - 1 or ep % 3 == 0:
                    net.eval()
                    with torch.no_grad(): pv = torch.cat([torch.softmax(net(Tva[i:i + 8192]), 1) for i in range(0, len(Tva), 8192)]).numpy()
                    from sklearn.metrics import f1_score
                    print(f"  fold {k} seed {sd} ep {ep} loss {tl_/len(Ttr):.4f} val row-F1 {f1_score(y[vas], pv.argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)", flush=True)
            net.eval()
            with torch.no_grad():
                pva += torch.cat([torch.softmax(net(Tva[i:i + 8192]), 1) for i in range(0, len(Tva), 8192)]).numpy() / a.seeds
                pte += torch.softmax(net(Tte), 1).numpy() / a.seeds
        oof[vas, val] = pva; test += pte / 5
        print(f"fold {k} done ({time.time()-t0:.0f}s)", flush=True)
    od = os.path.join(EXP, a.tag); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), oof); np.save(os.path.join(od, "test.npy"), test)
    cv["oof_f1_single"] = eval_single(oof, name=a.tag); cv["args"] = vars(a)
    if a.sim:
        s, _ = run_sim(os.path.join(od, "oof.npy"), a.tag); cv["sim"] = s; print("SIM raw", s.get("raw"), "chain", s.get("g_chain_cal_ps0.8"), flush=True)
    json.dump(cv, open(os.path.join(od, "res.json"), "w"), indent=1); print("done", f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
