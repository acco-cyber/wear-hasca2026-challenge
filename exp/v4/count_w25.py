"""Count-prior features from ALL FOUR LIMBS of every second (2nd-challenge test rows on test; the four sensor streams of
the training recordings on OOF), without grouping rows into seconds: for every (subject, exercise) the IMU-only class
probability mass per limb, the argmax count and the sorted-margin profile pooled over the subject's 4n rows.
Training subjects use the NESTED IMU classifier (w25run/ctxF/cache/imu_oof_s{s}.npy, fold-honest); test rows use the
full model (imu_full.txt) on the de-augmented 2025 rows (w25_clean.npz).
  python count_w25.py  ->  subs/xcount_w25.npz (oof_X, test_X; rows in profile_features key order)"""
import os, sys
import numpy as np, lightgbm as lgb
CTXF = r"E:\Claude code\wear\exp\v4\w25run\ctxF"
sys.path.insert(0, CTXF)
from ctxlib import tile_feats, lsm, PIPE_TO_25, CACHE
W = r"E:\Claude code\wear"
RK = [60, 70, 80, 90, 97, 105, 115, 125, 140]
SETS = {0: 2, 14: 2}
N_CLS = 19


def feats(LP, ns):
    """LP: list of 4 (n, 19) log-prob arrays (one per limb) -> (18, d) features for classes 1..18"""
    allp = np.concatenate(LP); P = np.exp(allp); out = []
    for c in range(1, N_CLS):
        f = []
        for lp in LP:
            p = np.exp(lp); f += [p[:, c].sum() / ns, (lp.argmax(1) == c).sum() / ns]
        sc = np.sort(allp[:, c] - np.delete(allp, c, 1).max(1))[::-1]
        f += [sc[min(int(r * 4 * ns), len(sc) - 1)] for r in RK]
        f += [P[:, c].sum() / (4 * ns), (allp.argmax(1) == c).sum() / (4 * ns), (sc > 0).sum() / (4 * ns), (sc > -1).sum() / (4 * ns),
              (sc > 1).sum() / (4 * ns)]
        out.append(f)
    return np.array(out, np.float64)


def main():
    st = np.load(os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4", "stage.npz"), allow_pickle=True)
    sbj = st["oof_sbj"].astype(int); tsbj = st["test_sbj"].astype(int)
    Xo = []
    for s in np.unique(sbj):
        A = np.load(os.path.join(CACHE, f"imu_oof_s{s}.npy")).astype(np.float64)          # (4, n, 19) nested log-probs
        assert A.shape[1] == (sbj == s).sum()
        Xo.append(feats([A[L] for L in range(4)], SETS.get(int(s), 1)))
    cz = np.load(os.path.join(CACHE, "w25_clean.npz")); acc, s25, l25 = cz["acc"], cz["sbj"], cz["limb"]
    mdl = lgb.Booster(model_file=os.path.join(CACHE, "imu_full.txt"))
    lp25 = np.zeros((len(acc), N_CLS))
    for L in range(4):
        r = np.flatnonzero(l25 == PIPE_TO_25[L])
        lp25[r] = lsm(mdl.predict(tile_feats(acc[r], L), raw_score=True, num_threads=4).astype(np.float64))
    Xt = []
    for s in np.unique(tsbj):
        LP = [lp25[(s25 == s) & (l25 == PIPE_TO_25[L])] for L in range(4)]
        assert all(len(x) == (tsbj == s).sum() for x in LP)
        Xt.append(feats(LP, 1))
    Xo, Xt = np.concatenate(Xo), np.concatenate(Xt)
    np.savez(os.path.join(W, "subs", "xcount_w25.npz"), oof_X=Xo, test_X=Xt)
    print("oof_X", Xo.shape, "test_X", Xt.shape, "| mean pooled mass per class (oof vs test):", Xo[:, 8 + 9].mean().round(1), Xt[:, 8 + 9].mean().round(1))


if __name__ == "__main__":
    main()
