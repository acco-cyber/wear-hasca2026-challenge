"""Nonparametric cross-limb compatibility: for window i (limb a) find its K nearest train seconds (limb a descriptor space,
fit sessions, own session excluded) and compare candidate j (limb b) with limb b's descriptors of those seconds at t+1 ("next")
and at t ("same"). Features: min / mean distance. -> feats/<s>_<r>_knn.npy (n,M,4)
python knnx.py"""
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
FD = os.path.join(XD, "feats")
DIMS = [DESC_NAMES.index(x) for x in ["ox", "oy", "oz", "lmag_mu", "lmag_sd", "fdom", "lb0", "lb1", "lb2", "lb3"]]
KNN_NAMES = ["kx_next_min", "kx_next_mean", "kx_same_min", "kx_same_mean"]
K = 20

class Bank:
    def __init__(self, imu, sl):
        Ds, sess, nxt = [], [], []
        for s in FIT:
            a, b = sl[s]; W = np.asarray(imu[a:b], np.float32)
            D = np.stack([window_desc(W[:, l])[:, DIMS] for l in range(4)], 1)          # (n,4,d)
            Ds.append(D); sess.append(np.full(len(D), FIT.index(s))); nxt.append(np.r_[np.arange(1, len(D)), -1])
        D = np.concatenate(Ds); self.sess = np.concatenate(sess); off = np.cumsum([0] + [len(x) for x in Ds])[:-1]
        self.nxt = np.concatenate([np.where(x >= 0, x + o, -1) for x, o in zip(nxt, off)])
        self.mu = np.nanmean(D.reshape(-1, D.shape[-1]), 0); self.sd = np.nanstd(D.reshape(-1, D.shape[-1]), 0) + 1e-3
        self.D = (D - self.mu) / self.sd
        vr = ~np.isnan(self.D).any(axis=(1, 2))
        self.ok = vr & (self.nxt >= 0) & vr[np.maximum(self.nxt, 0)]
        self.nn = {}
    def query(self, Z, a, excl):
        """Z (m,d) standardised descriptors of limb a; excl = fit-session index to exclude (or -1). Returns (m,K) bank rows."""
        key = (a, excl)
        if key not in self.nn:
            rows = np.where(self.ok & (self.sess != excl))[0]; self.nn[key] = (rows, NearestNeighbors(n_neighbors=K).fit(self.D[rows, a]))
        rows, nn = self.nn[key]; _, idx = nn.kneighbors(np.nan_to_num(Z)); return rows[idx]

def knn_feats(W, limb, cand, bank, excl=-1):
    n, M = cand.shape; Z = (window_desc(W)[:, DIMS] - bank.mu) / bank.sd; Z = np.nan_to_num(Z)
    NB = np.zeros((n, K), np.int64)
    for a in range(4):
        ia = np.where(limb == a)[0]
        if len(ia): NB[ia] = bank.query(Z[ia], a, excl)
    valid = cand >= 0; cc = np.where(valid, cand, 0); lb = limb[cc]                    # (n,M)
    out = np.full((n, M, 4), np.nan, np.float32)
    for kind, col in (("next", 0), ("same", 2)):
        rowsB = bank.nxt[NB] if kind == "next" else NB                                    # (n,K)
        for b in range(4):
            Ob = bank.D[rowsB, b]                                                         # (n,K,d)
            m = valid & (lb == b)
            ii, kk = np.nonzero(m)
            if len(ii) == 0: continue
            dist = np.sqrt(((Ob[ii] - Z[cc[ii, kk]][:, None, :]) ** 2).sum(2))           # (e,K)
            out[ii, kk, col] = dist.min(1); out[ii, kk, col + 1] = dist.mean(1)
    return out

if __name__ == "__main__":
    t0 = time.time(); meta, imu, vid, sl = load_prep(); bank = Bank(imu, sl); print("bank", bank.D.shape, f"{time.time()-t0:.0f}s", flush=True)
    for p in sorted(glob.glob(os.path.join(FD, "*.npz"))):
        q = p.replace(".npz", "_knn.npy")
        if os.path.exists(q): continue
        s = os.path.basename(p).rsplit("_", 1)[0]; d = np.load(p); cand, limb = d["cand"], d["limb"]; a, b = sl[s]
        W = session_windows(imu, a, b, limb); excl = FIT.index(s) if s in FIT else -1
        np.save(q, knn_feats(W, limb, cand, bank, excl)); print(os.path.basename(q), f"{time.time()-t0:.0f}s", flush=True)
