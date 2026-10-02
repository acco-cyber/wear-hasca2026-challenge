"""Fold-honest combined links (L2) for the 69,326 test-like OOF tiles of the train recordings.

Fork of woominyo's public notebook "WEAR@HASCA 2026 | Timeline Reconstruction + Graph" (Apache 2.0), which adapts
honghanhhh's "WEAR@HASCA Hungarian Chain Viterbi LB 0.74" (Apache 2.0). All data/link code below is copied verbatim
from that notebook; only the driver differs:
  * no window models: the OOF window blend log-probs (d["logp"]) and Plp come from our GPU fork's kept output
    (kernel koushikrudra/wear-hanbat-gpu, keep/blend.npz)
  * FIVE link scorers: scorer g is trained on the pair rows of the subjects of the other 4 folds (same rows as the
    notebook: true successor + 16 negatives, features of ridge x{h} for a row of fold h) and applied to all candidate
    pairs of the held-out fold-g subjects (features of ridge x{g}); Hungarian + cycle breaking per subject; scores are
    quantile-normalised per subject against the notebook's fixed QN_REF table (exactly as for test)
Output: /kaggle/working/oof_L2.npz  (succ = global OOF row or -1, score, score_qn, + diagnostics arrays)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import gc, hashlib, json, re, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb

SMOKE = os.environ.get("WEAR_SMOKE", "0") == "1"      # local code test on the subjects whose raw video is local
SMOKE_SUBJECTS = {int(x) for x in os.environ.get("WEAR_SMOKE_SUBJECTS", "5,9,7,8,6").split(",")}
SLUG = "3rd-wear-dataset-challenge-hasca-2026"
ON_KAGGLE = Path("/kaggle/input").exists()
GPU_OK = False                                        # CPU-only kernel: the Gram statistics use numpy


#@@BLOCK:83-92@@

INPUT = find_input()
WORK = Path(os.environ.get("WEAR_WORK", "/kaggle/working" if ON_KAGGLE else "wear_run")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
if os.environ.get("WEAR_ROOT"):
    ROOT = Path(os.environ["WEAR_ROOT"])
elif ON_KAGGLE and shutil.disk_usage("/tmp").free > 12e9:
    ROOT = Path("/tmp/wear")
else:
    ROOT = WORK / "_wear"
PROC = ROOT / "proc"
for _p in (ROOT, PROC):
    _p.mkdir(parents=True, exist_ok=True)
T_START, TIMES = time.time(), {}


#@@BLOCK:114-137@@


def find_keep():
    """the kept output folder of the GPU fork (kernel source); WEAR_KEEP overrides"""
    if os.environ.get("WEAR_KEEP"):
        return Path(os.environ["WEAR_KEEP"])
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "blend.npz" in files and "sim_meta.npz" in files:
            hits.append(Path(root))
        dirs[:] = [x for x in dirs if x not in ("train", "test", "inertial_feat", "videomae_feat")]
    print("kernel-source candidates:", hits)
    assert hits, "keep/blend.npz of koushikrudra/wear-hanbat-gpu not found under /kaggle/input"
    return hits[0]


KEEP = find_keep()
print(f"input {INPUT}\nwork {WORK}\nscratch {ROOT}\nkeep {KEEP}\nSMOKE={SMOKE} {sorted(SMOKE_SUBJECTS) if SMOKE else ''}",
      flush=True)
os.system("ls -la " + str(KEEP) if ON_KAGGLE else "echo")

#@@BLOCK:171-182@@


#@@BLOCK:273-279@@


#@@BLOCK:282-307@@
    return recs


with stage("W1 preprocessing (train only)"):
    if not (ROOT / ".done_prep").exists():
        recs = prep()
        (ROOT / ".done_prep").touch()
    recs = json.loads((PROC / "recordings.json").read_text())
    if not SMOKE:
        assert len(recs) == 24 and sum(r["T"] for r in recs) == 3466400 and sum(r["F"] for r in recs) == 2079840
        assert all(r["F"] * 5 == r["T"] * 3 for r in recs)
    print(f"{len(recs)} recordings, {len({r['sbj'] for r in recs})} subjects")

#@@BLOCK:343-405@@


#@@BLOCK:1193-1196@@


with stage("W2 test-like OOF windows"):
    ksim = {k: v.astype(np.int64) for k, v in dict(np.load(KEEP / "sim_meta.npz")).items()}
    _recs, _w = oof_windows()
    _w = {k: v.astype(np.int64) for k, v in _w.items()}
    if not SMOKE:
        sim = _w
        assert len(sim["y"]) == 69326
        assert np.bincount(sim["fold"]).tolist() == [14353, 12995, 14625, 12768, 14585]
        assert hashlib.sha1(sim["sensor"].tobytes()).hexdigest()[:12] == "f81d90085315"
        assert hashlib.sha1(np.stack([sim["rec"], sim["start"]]).tobytes()).hexdigest()[:12] == "e4177f2589d7"
        for k in ksim:
            assert (ksim[k] == sim[k]).all(), f"sim_meta[{k}] differs from the GPU fork"
        OOF_ROWS = np.arange(len(sim["y"]))
    else:
        # the subset's own fold split / limb draw differ from the full run: take the GPU fork's rows of these subjects
        stems_all = [c.stem for c in sorted((INPUT / "train" / "inertial_feat").glob("*.csv"))]
        loc = {r["stem"]: i for i, r in enumerate(recs)}
        OOF_ROWS = np.flatnonzero(np.isin(ksim["sbj"], sorted(SMOKE_SUBJECTS)))
        sim = {k: v[OOF_ROWS].copy() for k, v in ksim.items()}
        sim["rec"] = np.array([loc[stems_all[r]] for r in sim["rec"]], np.int64)
        a_ = set(zip(sim["rec"].tolist(), sim["start"].tolist()))
        b_ = set(zip(_w["rec"].tolist(), _w["start"].tolist()))
        assert a_ == b_, (len(a_), len(b_), len(a_ ^ b_))
        assert (np.sort(sim["y"]) == np.sort(_w["y"])).all()
    _acc, _vid = load_inputs(recs, sim)
    np.save(ROOT / "sim_vid.npy", _vid)
    np.save(ROOT / "sim_acc.npy", _acc)
    _kacc = np.load(KEEP / "sim_acc.npy", mmap_mode="r")[OOF_ROWS]
    print("sim_acc vs GPU fork: max abs diff", float(np.abs(_kacc - _acc).max()))
    assert np.allclose(_kacc, _acc, atol=1e-5)
    del _acc, _vid, _kacc, _w, ksim
    fold_of = {int(s): int(f) for s, f in zip(sim["sbj"], sim["fold"])}
    print("rows", len(sim["y"]), "folds", {f: sorted(s for s in fold_of if fold_of[s] == f) for f in range(FOLDS)})

with stage("load OOF window blend + Plp"):
    _bl = np.load(KEEP / "blend.npz")
    d = {"logp": _bl["oof_logp"][OOF_ROWS].astype(np.float32), "y": sim["y"], "sbj": sim["sbj"], "fold": sim["fold"],
         "sensor": sim["sensor"],
         "vid": lambda: np.load(ROOT / "sim_vid.npy", mmap_mode="r"),      # (N,15,768) float16, raw VideoMAE
         "acc": lambda: np.load(ROOT / "sim_acc.npy")}                     # (N,50,3) float32, g
    PLP_OOF = _bl["plp_oof"][OOF_ROWS].astype(np.float32)
    del _bl
    assert d["logp"].shape == (len(sim["y"]), N_CLS) and np.isfinite(d["logp"]).all()
    print(f"window blend OOF macro F1 {macro_f1(d['y'], d['logp'].argmax(1)):.4f} (GPU fork full: 0.7253-ish), "
          f"Plp {macro_f1(d['y'], PLP_OOF.argmax(1)):.4f}")
    TRUE_SUCC = true_successor(sim["rec"], sim["start"])
    N_OOF = len(d["y"])

#@@BLOCK:1515-1695@@

with stage("combined links A: ridge boundary predictors"):
    fold_sum = {}
    for g in range(FOLDS):
        acc_ = None
        for sb in sorted(s for s in fold_of if fold_of[s] == g):
            st_ = subject_stats([r for r in recs if r["sbj"] == sb])
            st_ = {k: np.array(v, np.float64) for k, v in st_.items()}
            if acc_ is None:
                acc_ = st_
            else:
                for k in acc_:
                    acc_[k] += st_[k]
            del st_
        if acc_ is None:                                        # SMOKE only: a fold without subjects
            continue
        fold_sum[g] = {k: (v.astype(np.float32) if k in ("S", "C") else v) for k, v in acc_.items()}
        del acc_
        print(f"fold {g}: statistics of {int(fold_sum[g]['n'])} tile pairs [{time.time() - T_START:.0f}s]", flush=True)
    RIDGES = {g: fit_ridge(fold_sum, [h for h in fold_sum if h != g]) for g in fold_sum}
    del fold_sum
    print("ridge r2 (forward, backward):", {g: (round(float(m["r2f"]), 3), round(float(m["r2b"]), 3)) for g, m in RIDGES.items()},
          "  (ours: about 0.58)")

#@@BLOCK:1719-2090@@

#@@BLOCK:2130-2228@@


with stage("combined links B: pair features of every train subject (ridge of its own held-out fold)"):
    _V, _A = d["vid"](), d["acc"]()
    SUB, hits = {}, []
    for s in np.unique(d["sbj"]):
        ii = np.flatnonzero(d["sbj"] == s)
        n = len(ii)
        g = fold_of[int(s)]
        cand, valid, base, new = subject_all(_V[ii], _A[ii], d["sensor"][ii], d["logp"][ii], PLP_OOF[ii], [RIDGES[g]])
        tl_ = np.full(N_OOF, -1)
        tl_[ii] = np.arange(n)
        true_loc = np.where(TRUE_SUCC[ii] >= 0, tl_[np.maximum(TRUE_SUCC[ii], 0)], -1)
        rp = np.repeat(np.arange(n), valid.sum(1))
        cj = cand[valid]
        pos = cj == true_loc[rp]
        rng = np.random.default_rng(7000 + int(s))
        key = rng.random(len(rp))
        key[pos] = -1.0                                       # the true successor first inside its row
        o = np.lexsort((key, rp))
        start = np.r_[0, np.cumsum(np.bincount(rp, minlength=n))]
        rank = np.arange(len(o)) - start[rp[o]]
        npos = np.bincount(rp, weights=pos.astype(float), minlength=n).astype(np.int64)
        keep = o[rank < LK_NEG + npos[rp[o]]]                 # true successor (if a candidate) + 16 random negatives
        hits.append(((cand == true_loc[:, None]) & valid).any(1)[true_loc >= 0])
        SUB[int(s)] = dict(ii=ii, g=g, cand=cand, valid=valid, base=base, new=new[0],
                           Xtr=np.concatenate([base[keep].astype(np.float32), new[0][keep]], 1),
                           Ttr=pos[keep].astype(np.float32))
        print(f"sbj {s} fold {g} n={n} K={cand.shape[1]} pairs {valid.sum()} recall {hits[-1].mean():.3f} "
              f"[{time.time() - T_START:.0f}s] RSS {rss_gb()[0]:.1f} GB", flush=True)
        del cand, valid, base, new, rp, cj, pos, key, o
    del _V, _A
    print(f"scorer rows {sum(len(v['Ttr']) for v in SUB.values())} positive rate "
          f"{np.concatenate([v['Ttr'] for v in SUB.values()]).mean():.4f} candidate recall {np.concatenate(hits).mean():.4f}"
          f"   (notebook, all 22 subjects: 1,173,696 / 0.0549 / 0.930)")


def diag(succ, m_rows, tag):
    m = (succ >= 0) & m_rows
    ex = (succ[m] == TRUE_SUCC[m]).mean()
    has_t = m_rows & (TRUE_SUCC >= 0)
    rec = (succ[has_t] == TRUE_SUCC[has_t]).mean()
    sl = (d["y"][succ[m]] == d["y"][m]).mean()
    print(f"{tag}: linked {m.sum() / max(m_rows.sum(), 1):.3f} exact {ex:.3f} (of rows with a true successor: {rec:.3f}) "
          f"same-label {sl:.3f}", flush=True)
    return dict(linked=float(m.sum() / max(m_rows.sum(), 1)), exact=float(ex), exact_of_true=float(rec), same=float(sl))


with stage("combined links C-E: 5 fold scorers -> held-out fold links"):
    SUCC, SCORE = np.full(N_OOF, -1, np.int64), np.full(N_OOF, -50.0, np.float32)
    TOPC, TOPL = np.full((N_OOF, 3), -1, np.int64), np.full((N_OOF, 3), -50.0, np.float32)
    DIAG, IMP = {}, {}
    folds_present = sorted({v["g"] for v in SUB.values()})
    for g in folds_present:
        tr = [s for s, v in SUB.items() if v["g"] != g]
        if not tr:
            print(f"fold {g}: no training subjects (SMOKE), skipped")
            continue
        X = np.concatenate([SUB[s]["Xtr"] for s in tr])
        T = np.concatenate([SUB[s]["Ttr"] for s in tr])
        t0 = time.time()
        model = lgb.train(LK_PARAMS, lgb.Dataset(X, T, free_raw_data=True), num_boost_round=LK_ROUNDS)
        print(f"fold {g}: scorer on {len(tr)} subjects, {len(T)} rows (pos {T.mean():.4f}) in {time.time() - t0:.0f}s",
              flush=True)
        IMP[g] = model.feature_importance("gain")
        del X, T
        for s in [s for s, v in SUB.items() if v["g"] == g]:
            v = SUB[s]
            ii, cand, valid = v["ii"], v["cand"], v["valid"]
            lo = np.empty(valid.sum(), np.float32)
            for c0 in range(0, len(lo), 1_000_000):
                Xp = np.concatenate([v["base"][c0:c0 + 1_000_000].astype(np.float32), v["new"][c0:c0 + 1_000_000]], 1)
                p_ = np.clip(model.predict(Xp, num_threads=4), 1e-6, 1 - 1e-6)
                lo[c0:c0 + len(p_)] = np.log(p_ / (1 - p_))
                del Xp
            L = np.full(cand.shape, -50.0, np.float32)
            L[valid] = lo
            s_loc, sc = lk_assign(cand, L)
            SUCC[ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1)
            SCORE[ii] = sc
            o = np.argsort(-L, 1)[:, :3]
            cl = np.take_along_axis(cand, o, 1)
            ll = np.take_along_axis(L, o, 1)
            okk = (cl >= 0) & (ll > -50.0)
            TOPC[ii] = np.where(okk, ii[np.maximum(cl, 0)], -1)
            TOPL[ii] = np.where(okk, ll, -50.0)
            m = SUCC[ii] >= 0
            print(f"  sbj {s} n={len(ii)} K={cand.shape[1]} linked {m.mean():.3f} "
                  f"exact {(SUCC[ii][m] == TRUE_SUCC[ii][m]).mean():.3f} median score {np.median(sc[m]):.2f} "
                  f"[{time.time() - T_START:.0f}s]", flush=True)
            del L, lo
        del model
        gc.collect()
        DIAG[g] = diag(SUCC, d["fold"] == g, f"fold {g}")
    _m = SUCC >= 0
    assert len(np.unique(SUCC[_m])) == _m.sum() and (d["sbj"][SUCC[_m]] == d["sbj"][_m]).all()
    SCORE_QN = qnorm(SUCC, SCORE, d["sbj"], QN_REF)
    DIAG["all"] = diag(SUCC, np.ones(N_OOF, bool), "L2 OOF all folds")
    # top-1 candidate (before the 1:1 assignment)
    m1 = TOPC[:, 0] >= 0
    print(f"top-1 candidate: exact {(TOPC[m1, 0] == TRUE_SUCC[m1]).mean():.3f}")
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("raw linked score quantiles  ", np.round(np.quantile(SCORE[_m], qs), 3).tolist())
    print("QN_REF quantiles (notebook) ", np.round(np.quantile(QN_REF, qs), 3).tolist())
    for s in np.unique(d["sbj"]):
        mm = (d["sbj"] == s) & _m
        print(f"  sbj {s} fold {fold_of[int(s)]}: linked {mm.sum() / (d['sbj'] == s).sum():.3f} "
              f"exact {(SUCC[mm] == TRUE_SUCC[mm]).mean():.3f} raw median {np.median(SCORE[mm]):.2f}")
    # L0 links of the GPU fork, same rows, for reference
    l0 = np.load(KEEP / "links_L0.npz")
    if not SMOKE:
        DIAG["L0"] = diag(l0["oof_succ"].astype(np.int64), np.ones(N_OOF, bool), "L0 OOF (GPU fork) for reference")
    names = BASE49 + NEW16
    imp = np.mean([IMP[g] / IMP[g].sum() for g in IMP], 0)
    print("top features (gain share):", [(names[i], round(float(imp[i]), 3)) for i in np.argsort(-imp)[:15]])
    np.savez(WORK / "oof_L2.npz", succ=SUCC, score=SCORE, score_qn=SCORE_QN, top_cand=TOPC, top_lo=TOPL,
             true_succ=TRUE_SUCC, rows=OOF_ROWS, sbj=d["sbj"], fold=d["fold"], y=d["y"])
    (WORK / "oof_L2_diag.json").write_text(json.dumps(DIAG, indent=1))
    print("saved", WORK / "oof_L2.npz")

print("stage times (min):", {k: round(v / 60, 1) for k, v in TIMES.items()})
print(f"done in {(time.time() - T_START) / 60:.1f} min")
