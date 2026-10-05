"""Label-free per-tile features for the null-edge specialist, built for OOF and test from the refined fused labels and
all 16 bagged matchings of K7 + K9. Saves feats.npz (X_o, X_t, cand masks, feature names)."""
import os, time
import numpy as np
from ne_common import load_all, walks, prev_of, HERE
from v4_local import N_CLS

HMAX = 4          # walk depth for change distances
T0 = time.time()


def lo(M, c):
    """log-odds class c vs best other class of a probability matrix"""
    M = np.clip(M.astype(np.float64), 1e-9, None)
    oth = M.copy(); oth[np.arange(len(M)), c] = -1
    return np.log(M[np.arange(len(M)), c]) - np.log(oth.max(1))


def null_feats(M, prefix, isprob=True):
    """null prob/logp, top activity prob, null-vs-activity log-odds"""
    M = M.astype(np.float64)
    if not isprob:
        M = np.exp(M - M.max(1, keepdims=True)); M /= M.sum(1, keepdims=True)
    M = np.clip(M, 1e-9, None)
    n0 = M[:, 0]; ta = M[:, 1:].max(1)
    return {f"{prefix}_n": n0, f"{prefix}_ta": ta, f"{prefix}_lo": np.log(n0) - np.log(ta)}


def build(split, d):
    o = split == "oof"
    lab = d["ref_o"] if o else d["ref_t"]; fin = d["fin_o"] if o else d["fin_t"]
    sb = d["oof_sbj"] if o else d["test_sbj"]
    P, Q, B, lw = (d["Po"], d["Qo"], d["Bo"], d["lwo"]) if o else (d["Pt"], d["Qt"], d["Bt"], d["lwt"])
    L, Ls = (d["Lo"], d["Lso"]) if o else (d["Lt"], d["Lst"])
    pre = "oof_" if o else "test_"
    ener, post, vmot, vmean = d[pre + "ener"], d[pre + "post"], d[pre + "vmot"], d[pre + "vmean"].astype(np.float32)
    sens = d["sensor_oof"] if o else d["sensor_test"]
    n = len(lab); ar = np.arange(n)
    f = {}
    f.update(null_feats(P, "P")); f.update(null_feats(Q, "Q"))
    f["P_lab"] = P[ar, lab]; f["Q_lab"] = Q[ar, lab]
    f["P_lo_lab"] = lo(P, lab); f["Q_lo_lab"] = lo(Q, lab)
    qa = Q[:, 1:].argmax(1) + 1
    f["Q_lo_qa"] = lo(Q, qa); f["qa_eq_lab"] = (qa == lab).astype(float)
    for t in ("7", "9"):
        f.update(null_feats(d[("P%so" if o else "P%st") % t], "P" + t))
        f.update(null_feats(d[("B2_OOF_" if o else "B2_TEST_") + t], "B" + t, False))
        f.update(null_feats(d[("oof_logp_" if o else "test_logp_") + t], "W" + t, False))
        f.update(null_feats(d[("TA_OOF_" if o else "TA_TEST_") + t], "TA" + t, True))
        f.update(null_feats(d[("tab_S3_oof_" if o else "tab_S3_test_") + t], "S3" + t, True))
        f.update(null_feats(d[("QB_OOF_" if o else "QB_TEST_") + t], "QB" + t, True))
    f.update(null_feats(B, "B", False)); f.update(null_feats(lw, "W", False))
    f["ener"], f["vmot"] = ener.astype(np.float64), vmot.astype(np.float64)
    for k in range(post.shape[1]):
        f[f"post{k}"] = post[:, k].astype(np.float64)
    f["sens"] = sens.astype(float)
    f["lab_null"] = (lab == 0).astype(float); f["fin_null"] = (fin == 0).astype(float); f["refined"] = (lab != fin).astype(float)
    # subject-level context (label-free)
    f["sbj_null_frac"] = np.zeros(n); f["sbj_n"] = np.zeros(n); f["sbj_P0_mean"] = np.zeros(n); f["lab_frac"] = np.zeros(n)
    for s in np.unique(sb):
        ii = sb == s
        f["sbj_null_frac"][ii] = (lab[ii] == 0).mean(); f["sbj_n"][ii] = ii.sum(); f["sbj_P0_mean"][ii] = P[ii, 0].mean()
        bc = np.bincount(lab[ii], minlength=N_CLS) / ii.sum(); f["lab_frac"][ii] = bc[lab[ii]]
    # refiner probability mass per tile
    tl, ot, pr = (d["r_tl"], d["r_ot"], d["r_pr"]) if o else (d["rt_tl"], d["rt_ot"], d["rt_pr"])
    K = L.shape[0]
    rn = np.zeros(n); ra = np.zeros(n); rmax = np.zeros(n)
    np.add.at(rn, tl[ot == 0], pr[ot == 0] / K); np.add.at(ra, tl[ot != 0], pr[ot != 0] / K); np.maximum.at(rmax, tl, pr)
    f["ref_to_null"], f["ref_to_act"], f["ref_max"] = rn, ra, rmax
    # chain features over all matchings
    Pn = f["P_lo"]; Qn = f["Q_lo"]
    M = L.shape[0]
    df_, db_, endf, endb = np.zeros((M, n)), np.zeros((M, n)), np.zeros((M, n)), np.zeros((M, n))
    nb_null = {k: np.zeros((M, n)) for k in (-3, -2, -1, 1, 2, 3)}
    nb_same = {k: np.zeros((M, n)) for k in (-2, -1, 1, 2)}
    nb_P = {k: np.zeros((M, n)) for k in (-3, -2, -1, 1, 2, 3)}
    nb_Q = {k: np.zeros((M, n)) for k in (-1, 1)}
    nb_E = {k: np.zeros((M, n)) for k in (-1, 1)}
    vd = {k: np.zeros((M, n)) for k in (-1, 1)}
    chg_dist = np.zeros((M, n))            # nearest change (either direction) in steps, HMAX+1 if none
    for m in range(M):
        Fw, Bw = walks(L[m], HMAX)
        for A, dist, end in ((Fw, df_, endf), (Bw, db_, endb)):
            dcur = np.full(n, HMAX + 1.0); e = np.full(n, HMAX + 1.0); done = np.zeros(n, bool)
            for k in range(HMAX):
                g = A[k]; ok = g >= 0
                ch = ok & (lab[np.maximum(g, 0)] != lab) & ~done
                dcur[ch] = k + 1; done |= ch
                en = ~ok & (e > HMAX)
                e[en] = k + 1
                done |= ~ok
            dist[m] = dcur; end[m] = e
        chg_dist[m] = np.minimum(df_[m], db_[m])
        for k in (1, 2, 3):
            for sgn, A in ((1, Fw), (-1, Bw)):
                g = A[k - 1]; ok = g >= 0; gg = np.maximum(g, 0)
                nb_null[sgn * k][m] = np.where(ok, (lab[gg] == 0).astype(float), np.nan)
                nb_P[sgn * k][m] = np.where(ok, Pn[gg], np.nan)
                if k <= 2:
                    nb_same[sgn * k][m] = np.where(ok, (lab[gg] == lab).astype(float), np.nan)
                if k == 1:
                    nb_Q[sgn][m] = np.where(ok, Qn[gg], np.nan)
                    nb_E[sgn][m] = np.where(ok, ener[gg] - ener, np.nan)
                    vd[sgn][m] = np.where(ok, np.linalg.norm(vmean[gg] - vmean, axis=1), np.nan)
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    nanm = lambda A: np.nanmean(A, 0)
    f["df_mean"], f["db_mean"] = df_.mean(0), db_.mean(0); f["df_min"], f["db_min"] = df_.min(0), db_.min(0)
    f["chg_min"], f["chg_mean"] = chg_dist.min(0), chg_dist.mean(0); f["chg_frac1"] = (chg_dist <= 1).mean(0)
    f["endf_mean"], f["endb_mean"] = endf.mean(0), endb.mean(0)
    for k, A in nb_null.items():
        f[f"nbnull{k}"] = nanm(A)
    for k, A in nb_same.items():
        f[f"nbsame{k}"] = nanm(A)
    for k, A in nb_P.items():
        f[f"nbP{k}"] = nanm(A)
    for k, A in nb_Q.items():
        f[f"nbQ{k}"] = nanm(A)
    for k, A in nb_E.items():
        f[f"nbE{k}"] = nanm(A)
    for k, A in vd.items():
        f[f"vd{k}"] = nanm(A)
    f["nbP_sum3"] = np.nansum([f[f"nbP{k}"] for k in (-3, -2, -1, 1, 2, 3)], 0)
    f["nbnull_sum3"] = np.nansum([f[f"nbnull{k}"] for k in (-3, -2, -1, 1, 2, 3)], 0)
    # agreement across matchings: modal successor / predecessor share, link scores
    PR = np.stack([prev_of(L[m]) for m in range(M)])
    for nm, A in (("succ", L), ("prev", PR)):
        As = np.sort(A, 0); best = np.ones(n); run = np.ones(n)
        for m in range(1, M):
            same = As[m] == As[m - 1]; run = np.where(same, run + 1, 1); best = np.maximum(best, run)
        f[f"{nm}_agree"] = best / M; f[f"{nm}_exist"] = (A >= 0).mean(0)
    f["score_mean"] = Ls.mean(0)
    sc_in = np.zeros((M, n))
    for m in range(M):
        pv = PR[m]; sc_in[m] = np.where(pv >= 0, Ls[m][np.maximum(pv, 0)], np.nan)
    f["score_in"] = nanm(sc_in)
    # candidate masks
    cand3 = chg_dist.min(0) <= 4          # change within <= 3 steps => tile is <= 3 steps from a changing pair (dist <= 4 walk steps)
    cand_any = (chg_dist.min(0) <= HMAX) | (endf.min(0) <= 1) | (endb.min(0) <= 1)
    names = list(f.keys())
    X = np.stack([np.asarray(f[k], np.float64) for k in names], 1).astype(np.float32)
    print(f"[{time.time() - T0:.0f}s] {split}: {X.shape}, cand3 {cand3.sum()}, cand_any {cand_any.sum()}", flush=True)
    return X, names, chg_dist.min(0), endf.min(0), endb.min(0), qa


def main():
    d = load_all()
    Xo, names, cdo, efo, ebo, qao = build("oof", d)
    Xt, _, cdt, eft, ebt, qat = build("test", d)
    np.savez_compressed(os.path.join(HERE, "feats.npz"), Xo=Xo, Xt=Xt, names=np.array(names), cdo=cdo, cdt=cdt,
                        efo=efo, ebo=ebo, eft=eft, ebt=ebt, qao=qao, qat=qat)
    print("saved", len(names), "features")


if __name__ == "__main__":
    main()
