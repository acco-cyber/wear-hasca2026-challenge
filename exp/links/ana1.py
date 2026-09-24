"""Baseline reproduction + rank of the true successor under candidate descriptors (eval sessions, sim limbs)."""
import time
from lk import *

S = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb"))
meta, imu, vid = load_prep(); oof = blend_oof(0.2)
t0 = time.time()
rows = []
for s in EVAL:
    st = S[s]; n = st["n"]; y = st["y"]; P = sim_P(oof, st)
    lab = decode(P, st["cand"], st["lo"], st["succ0"], st["sc"], st["Lm"])
    m = link_metrics(cut(st["succ0"], st["sc"], st["Lm"], -6.0), y, st["cand"])
    rows.append(dict(session=s, f1=f1(y, lab), **m))
print(pd.DataFrame(rows).round(4).to_string()); print("mean", pd.DataFrame(rows).mean(numeric_only=True).round(4).to_dict(), f"{time.time()-t0:.0f}s")

def nrm(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def rank_true(S_):
    n = len(S_); S_ = S_.copy(); np.fill_diagonal(S_, -np.inf)
    tr = S_[np.arange(n - 1), np.arange(1, n)]
    return (S_[:-1] > tr[:, None]).sum(1)  # 0 = best

res = []
for s in EVAL:
    a, b = S[s]["a"], S[s]["b"]; V = np.asarray(vid[a:b], np.float32); n = len(V)
    Vn = nrm(V)
    D = {}
    D["th3"] = nrm(Vn[:, -3:].mean(1)) @ nrm(Vn[:, :3].mean(1)).T
    D["11"] = Vn[:, -1] @ Vn[:, 0].T
    D["mp"] = nrm(Vn.mean(1)) @ nrm(Vn.mean(1)).T
    # raw-space extrapolation
    for kk in (15, 8, 4):
        pos = np.arange(15 - kk, 15, dtype=np.float32); pc = pos - pos.mean()
        slope = (V[:, 15 - kk:] * pc[None, :, None]).sum(1) / (pc ** 2).sum()
        cen = V[:, 15 - kk:].mean(1)
        for h in (8, 16):
            pred = cen + slope * (14 + h - pos.mean())   # predicted value at position 14+h
            D[f"ex{kk}_h{h}"] = nrm(pred) @ Vn[:, 0].T
        # backward from b
        posb = np.arange(0, kk, dtype=np.float32); pcb = posb - posb.mean()
        slb = (V[:, :kk] * pcb[None, :, None]).sum(1) / (pcb ** 2).sum(); cb = V[:, :kk].mean(1)
        predb = cb + slb * (-16 - posb.mean())
        D[f"exb{kk}"] = Vn[:, -1] @ nrm(predb).T
        predf = cen + slope * (30 - pos.mean())
        D[f"exfb{kk}"] = D[f"exb{kk}"] + nrm(predf) @ Vn[:, 0].T
    # euclidean in raw space
    last = V[:, -1]; first = V[:, 0]
    D["neg_euc11"] = -(np.sum(last ** 2, 1)[:, None] + np.sum(first ** 2, 1)[None] - 2 * last @ first.T)
    for k_, M in D.items():
        r = rank_true(M)
        res.append(dict(session=s, desc=k_, top1=(r < 1).mean(), top10=(r < 10).mean(), top40=(r < 40).mean(), top100=(r < 100).mean(), med=np.median(r)))
df = pd.DataFrame(res); print(df.groupby("desc").mean(numeric_only=True).sort_values("top10", ascending=False).round(3).to_string())
