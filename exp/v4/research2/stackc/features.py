"""Stage-C features: per (tile, candidate class) evidence from the fused b4wa decode, both kept fits (K7, K9), the
chain neighbours along all 16 bagged matchings and the per-subject whitened video kNN. Label-free for every tile
(uses fused / refined labels, never oof_y), identical recipe for OOF and test.
  python features.py   -> cache/feat.npz"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
from v4_local import whitened_subject, unit, macro_f1

W = r"E:\Claude code\wear"
K7 = W + r"\work\v4\wear-v4-big-pool-opt-s7\keep4"
K9 = W + r"\work\v4\wear-v4-big-tf-opt-s9\keep4"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
KC = 4          # candidates per tile
NC = 19
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def rel(L, C):
    """log-odds of candidate c vs the best other class, from a log-prob matrix L (n,19) and candidates C (n,K)"""
    Lc = np.take_along_axis(L, C, 1)
    s = np.sort(L, 1); top1, top2 = s[:, -1:], s[:, -2:-1]
    best_other = np.where(Lc >= top1, top2, top1)
    return (Lc - best_other).astype(np.float32)


def steps(succ_list, n, dmax):
    """forward / backward index arrays per matching and distance: (M, dmax, n)"""
    M = len(succ_list); F = np.full((M, dmax, n), -1, np.int64); B = np.full((M, dmax, n), -1, np.int64)
    for m, su in enumerate(succ_list):
        pr = np.full(n, -1); ok = su >= 0; pr[su[ok]] = np.flatnonzero(ok)
        f = np.arange(n); b = np.arange(n)
        for d in range(dmax):
            f = np.where(f >= 0, su[np.maximum(f, 0)], -1); b = np.where(b >= 0, pr[np.maximum(b, 0)], -1)
            F[m, d], B[m, d] = f, b
    return F, B


def build(split, Q, lab, fits, sbj, emb):
    n = len(lab); o = "oof" if split == "o" else "test"; up = "OOF" if split == "o" else "TEST"
    lQ = np.log(np.clip(Q, 1e-9, None))
    rk = np.argsort(-Q, 1)
    C = rk[:, :KC].copy()
    miss = ~(C == lab[:, None]).any(1); C[miss, KC - 1] = lab[miss]
    G, gn = [], []
    s = np.sort(Q, 1)
    ent = -(Q * lQ).sum(1)
    G += [s[:, -1], s[:, -1] - s[:, -2], ent, (lab != Q.argmax(1)).astype(np.float32)]; gn += ["conf", "margin", "entropy", "flipped"]
    # per-subject confidence rank
    cr = np.zeros(n, np.float32)
    for sb in np.unique(sbj):
        ii = np.flatnonzero(sbj == sb); cr[ii] = np.argsort(np.argsort(s[ii, -1])) / len(ii)
    G += [cr]; gn += ["conf_rank"]
    F, fn = [], []
    F += [C.astype(np.float32), (np.argsort(rk, 1)[np.arange(n)[:, None], C]).astype(np.float32), np.take_along_axis(lQ, C, 1), rel(lQ, C),
          (C == lab[:, None]).astype(np.float32)]
    fn += ["cls", "rankQ", "logQ", "relQ", "is_lab"]
    srcs = []
    for nm, f in zip(("k7", "k9"), fits):
        srcs += [(nm + "_PB", np.log(np.clip(f["PB_" + up].astype(np.float64), 1e-9, None))),
                 (nm + "_QB", np.log(np.clip(f["QB_" + up].astype(np.float64), 1e-9, None)))]
    avg = lambda key: lsm(np.mean([f[key].astype(np.float64) for f in fits], 0))
    srcs += [("B2", avg("B2_" + up)), ("S3", avg("tab_S3_" + o)), ("TA", avg("TA_" + up)), ("win", avg(o + "_logp"))]
    for nm, L in srcs:
        F.append(rel(L, C)); fn.append("rel_" + nm)
    log(f"{split}: base features done")
    # scalars
    sc = [np.load(os.path.join(K7, "tile_scalars.npz"))[f"{o}_{k}"] for k in ("ener", "vmot")]
    post = np.load(os.path.join(K7, "tile_scalars.npz"))[f"{o}_post"]
    sens = fits[0]["sensor_" + o].astype(np.float32)
    G += [sc[0], sc[1], post[:, 0], post[:, 1], post[:, 2], sens]; gn += ["ener", "vmot", "post0", "post1", "post2", "sensor"]
    # chain neighbours along the 16 matchings
    succ = [f[o + "_succ"][k].astype(np.int64) for f in fits for k in range(8)]
    score = [f[o + "_score"][k].astype(np.float32) for f in fits for k in range(8)]
    M = len(succ); DM = 5
    Fw, Bw = steps(succ, n, DM)
    G += [np.mean([sc_ * (su >= 0) for sc_, su in zip(score, succ)], 0), (Fw[:, 0] >= 0).mean(0), (Bw[:, 0] >= 0).mean(0)]
    gn += ["lscore", "has_succ", "has_pred"]
    labx = np.r_[lab, -1]; Qx = np.vstack([Q, np.zeros((1, NC))]); lQx = np.vstack([lQ, np.zeros((1, NC))])
    for nm, A in (("f", Fw), ("b", Bw)):
        for d in range(3):
            idx = A[:, d]                                       # (M, n)
            F.append(np.mean(labx[idx][:, :, None] == C[None], 0).astype(np.float32)); fn.append(f"agree_{nm}{d + 1}")
        idx = A[:, 0]
        F.append(np.mean(Qx[idx[:, :, None], C[None]], 0).astype(np.float32)); fn.append(f"Q_{nm}1")
        F.append(np.mean(np.stack([rel(lQx[idx[m]], C) * (idx[m] >= 0)[:, None] for m in range(M)]), 0)); fn.append(f"relQ_{nm}1")
        # window mean of Q[c] over distances 1..5
        F.append(np.mean([Qx[A[:, d][:, :, None], C[None]].mean(0) for d in range(DM)], 0).astype(np.float32)); fn.append(f"Qwin_{nm}5")
        # run length of the candidate label (capped 20) along each matching, averaged
        run = np.zeros((n, KC), np.float32)
        for m in range(M):
            cur = np.arange(n); alive = np.ones((n, KC), bool); r = np.zeros((n, KC), np.float32)
            su = succ[m] if nm == "f" else None
            if nm == "b":
                pr = np.full(n, -1); ok = succ[m] >= 0; pr[succ[m][ok]] = np.flatnonzero(ok); su = pr
            for _ in range(20):
                cur = np.where(cur >= 0, su[np.maximum(cur, 0)], -1)
                alive &= (labx[cur][:, None] == C); r += alive
            run += r / M
        F.append(run); fn.append(f"run_{nm}")
    log(f"{split}: chain features done")
    # per-subject whitened video kNN
    Ew = unit(whitened_subject(emb, sbj, 0.5, 0.1))
    kf10, kq10, kf30, ksim = np.zeros((n, KC), np.float32), np.zeros((n, KC), np.float32), np.zeros((n, KC), np.float32), np.zeros(n, np.float32)
    for sb in np.unique(sbj):
        ii = np.flatnonzero(sbj == sb); S = Ew[ii] @ Ew[ii].T; np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, 30, axis=1)[:, :30]; v = np.take_along_axis(S, nb, 1); oo = np.argsort(-v, 1)
        nb, v = np.take_along_axis(nb, oo, 1), np.take_along_axis(v, oo, 1); del S
        g = ii[nb]                                              # global ids (ns, 30)
        Cs = C[ii]
        L10 = lab[g[:, :10]]; L30 = lab[g]
        kf10[ii] = (L10[:, :, None] == Cs[:, None]).mean(1); kf30[ii] = (L30[:, :, None] == Cs[:, None]).mean(1)
        w = np.exp((v[:, :10] - v[:, :1]) / 0.1); w /= w.sum(1, keepdims=True)
        kq10[ii] = np.einsum("nk,nkc->nc", w, Q[g[:, :10][:, :, None], Cs[:, None]])
        ksim[ii] = v[:, :10].mean(1)
    F += [kf10, kq10, kf30]; fn += ["knn_f10", "knn_q10", "knn_f30"]; G += [ksim]; gn += ["knn_sim"]
    log(f"{split}: kNN features done")
    return C, np.stack(F, 2).astype(np.float32), np.stack(G, 1).astype(np.float32), fn, gn


def main():
    os.makedirs(OUT, exist_ok=True)
    fits = []
    for d in (K7, K9):
        st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True); lk = np.load(os.path.join(d, "links.npz"))
        f = {k: st[k] for k in st.files}; f.update({k: lk[k] for k in ("oof_succ", "oof_score", "test_succ", "test_score")}); fits.append(f)
    y = fits[0]["oof_y"].astype(int); sbj = fits[0]["oof_sbj"].astype(int); tsbj = fits[0]["test_sbj"].astype(int)
    Qo = np.load(W + r"\subs\sub_v4c_b4wa_Qo.npy").astype(np.float64); Qt = np.load(W + r"\subs\sub_v4c_b4wa_Qt.npy").astype(np.float64)
    lo = np.load(W + r"\subs\sub_v4c_b4wa_labo.npy").astype(int); lt = np.load(W + r"\subs\sub_v4c_b4wa_labt.npy").astype(int)
    eo = np.load(os.path.join(K7, "oof_emb.npy")).astype(np.float32); et = np.load(os.path.join(K7, "test_emb.npy")).astype(np.float32)
    Co, Fo, Go, fn, gn = build("o", Qo, lo, fits, sbj, eo)
    Ct, Ft, Gt, _, _ = build("t", Qt, lt, fits, tsbj, et)
    log(f"truth in candidates: {(Co == y[:, None]).any(1).mean():.4f}; features {len(fn)} per candidate, {len(gn)} per tile")
    np.savez(os.path.join(OUT, "feat.npz"), Co=Co, Fo=Fo, Go=Go, Ct=Ct, Ft=Ft, Gt=Gt, fn=np.array(fn), gn=np.array(gn),
             y=y, sbj=sbj, fold=fits[0]["oof_fold"].astype(int), tsbj=tsbj, Qo=Qo.astype(np.float32), Qt=Qt.astype(np.float32), lo=lo, lt=lt)
    log("saved")


if __name__ == "__main__":
    main()
