"""baseline link quality on the OOF: kernel matchings of K7/K9, L3 candidate recall, plain Hungarian on L3 log-odds,
split by same-limb vs cross-limb true links"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(r"E:\Claude code\wear", "exp", "v4"))
from relink import lk_assign, fuse_subject
from common import *
S = load_stage(); sbj = S["oof_sbj"]; ts = S["true_succ"]; sens = S["sensor_oof"]
for d in (K7, K9):
    lk = np.load(os.path.join(d, "links.npz")); s0 = lk["oof_succ"]
    h = ts >= 0
    print(os.path.basename(os.path.dirname(d)), "kernel matchings exact:", [round(float(np.mean(s0[k][h] == ts[h])), 4) for k in range(len(s0))])
    same = h & (sens == sens[np.maximum(ts, 0)])
    print("   plain matching exact on same-limb true links", round(float(np.mean(s0[0][same] == ts[same])), 4), "cross-limb", round(float(np.mean(s0[0][h & ~same] == ts[h & ~same])), 4), "share same", round(float(same.sum() / h.sum()), 3))
Z = [np.load(os.path.join(d, "link_logodds.npz")) for d in (K7, K9)]
rec, ex7, exf, n_ = [], [], [], 0
for s in np.unique(sbj):
    ii = np.flatnonzero(sbj == s); pos = np.full(len(sbj), -1); pos[ii] = np.arange(len(ii))
    tl = np.where(ts[ii] >= 0, pos[np.maximum(ts[ii], 0)], -1); h = tl >= 0
    c7, L7 = Z[0][f"oof_{s}_cand"].astype(np.int64), Z[0][f"oof_{s}_L"]
    rec.append(((c7 == tl[:, None]) & (L7 > -49)).any(1)[h])
    su, _ = lk_assign(c7, L7); ex7.append(su[h] == tl[h])
    cf, Lf = fuse_subject([z[f"oof_{s}_cand"].astype(np.int64) for z in Z], [z[f"oof_{s}_L"] for z in Z], np.array([0.5, 0.5]))
    su, _ = lk_assign(cf, Lf); exf.append(su[h] == tl[h])
print("L3 candidate recall (K7)", np.concatenate(rec).mean().round(4), "| plain Hungarian on K7 log-odds exact", np.concatenate(ex7).mean().round(4),
      "| on K7+K9 fused log-odds", np.concatenate(exf).mean().round(4))
