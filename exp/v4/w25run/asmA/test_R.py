"""asmA variant R on the REAL test: assembly of the 2025 limb chains (testdeaug_chain_s22..25) with votes from K7's
chain-rescored test links (links_test_K7_deaug) + top-3 L3 test candidates, then the stage-2 test log-odds (testD full
models, world_table/score_world port) shifted by +/-delta by assembly agreement / contradiction, Hungarian + 7 perturbed
members, qnorm. Config = the combo chosen on ALL training subjects (same criterion as the nested choice).
  python test_R.py --cfg t0.9_m0.5_c1 --gmin 10 --delta 2 --oof links_asmA_R.npz --out links_asmA_R_test.npz"""
import os, sys, time, json, argparse
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TD = r"E:\Claude code\wear\exp\v4\w25run\testD"
sys.path.insert(0, TD)
_argv = sys.argv; sys.argv = [_argv[0]]
from tdlib import K7, chain_tuple, topk_cands, subject_table, match, qnorm, gap_feats, feat_cols, GAP_NAMES, SENS   # noqa: E402
sys.argv = _argv
from asmlib import build_votes, Assembler                     # noqa: E402
import lightgbm as lgb                                         # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--gmin", type=int, required=True)
ap.add_argument("--delta", type=float, required=True); ap.add_argument("--oof", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--lam", type=float, default=0.25)
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:.0f}s]", *x, flush=True)


thr = float(a.cfg.split("_")[0][1:]); mmin = float(a.cfg.split("_")[1][1:]); cmin = int(a.cfg.split("_")[2][1:])
st = np.load(os.path.join(K7, "stage.npz"), allow_pickle=True); tsbj = st["test_sbj"].astype(np.int64); tsens = st["sensor_test"].astype(np.int64); NT = len(tsbj)
LT = np.load(os.path.join(TD, "links_test_K7_deaug.npz")); TSU = LT["test_succ"].astype(np.int64); TSC = LT["test_score"].astype(np.float32)
LZ = np.load(os.path.join(K7, "link_logodds.npz"))
REF = np.load(os.path.join(K7, "links.npz"))["qn_ref"]
M1 = lgb.Booster(model_file=os.path.join(TD, "models", "K7_deaug_s1_full.txt")); M2 = lgb.Booster(model_file=os.path.join(TD, "models", "K7_deaug_s2_full.txt"))
OutSu = np.full((8, NT), -1, np.int64); OutSc = np.full((8, NT), -50.0, np.float32)
info = {}
for s in (22, 23, 24, 25):
    cf = np.load(os.path.join(TD, "cache", f"testdeaug_chain_s{s}.npz"))
    loc = np.flatnonzero(tsbj == s); n = len(loc); l_of = np.full(NT, -1); l_of[loc] = np.arange(n)
    assert (cf["loc"] == loc).all() and (cf["lim"] == tsens[loc]).all()
    lim = cf["lim"].astype(np.int64); anch = cf["anch"].astype(bool); pos0 = cf["pos"].astype(np.int64)
    # ---- world_table (build_test_links port, real test: no thinning)
    pos = pos0.copy(); pos[~anch] = 0
    owner = cf["owner"].astype(np.int64).copy(); keep = np.zeros_like(owner, bool); ai = np.flatnonzero(anch); keep[lim[ai], pos0[ai]] = True
    owner = np.where(keep, owner, -1); assert (owner >= 0).sum() == anch.sum()
    chains = [chain_tuple(cf[f"succ{L}"], cf[f"conf{L}"]) for L in range(4)]
    cand, Lo = topk_cands(LZ[f"test_{s}_cand"], LZ[f"test_{s}_L"], 50)
    t = subject_table(s, n, lim, anch, pos, owner, chains, cand, Lo, np.full(n, -1), 16)
    names = t["names"]; C1 = feat_cols(names); C2 = feat_cols(names + GAP_NAMES)
    base = t["X"][:, names.index("lo")].astype(np.float64)
    p1 = (M1.predict(t["X"][:, C1], raw_score=True) + base).astype(np.float32)
    _, r1 = match(t, p1, 1, s); g = gap_feats(t, r1[0][0], 16)
    Xk = np.concatenate([t["X"], g, p1[:, None]], 1)
    p2 = (M2.predict(Xk[:, C2], raw_score=True) + p1).astype(np.float64).astype(np.float32)
    _, r2 = match(t, p2, 1, s)
    stored = np.where(TSU[0][loc] >= 0, l_of[np.maximum(TSU[0][loc], 0)], -1)
    log(f"sbj {s}: harness stage-2 test member 0 reproduced {np.mean(r2[0][0] == stored):.4f}")
    # ---- assembly on the real 2025 chains (node ids are 2025-row-based, no time order)
    msu = np.where(TSU[:, loc] >= 0, l_of[np.maximum(TSU[:, loc], 0)], -1); msc = TSC[:, loc]
    vi, vj, vw = build_votes(n, msu, msc, LZ[f"test_{s}_cand"].astype(np.int64), LZ[f"test_{s}_L"].astype(np.float32), a.lam)
    rng = np.random.default_rng(9000 + s); tp = rng.permutation(n); inv = np.argsort(tp)
    lim_n = np.empty(n, np.int64); lim_n[tp] = lim; anch_n = np.empty(n, bool); anch_n[tp] = anch; pos_n = np.empty(n, np.int64); pos_n[tp] = np.where(anch, pos0, -1)
    A = Assembler(n, lim_n, anch_n, pos_n, [cf[f"succ{L}"] for L in range(4)], [cf[f"conf{L}"] for L in range(4)], thr)
    hist = A.run(tp[vi], tp[vj], vw, mmin, cmin, 12, True); r = A.result()
    ra = r["asu"][tp]; asu = np.where(ra >= 0, inv[np.maximum(ra, 0)], -1); root = r["root"][tp]; coord = r["coord"][tp]; gt = r["gtiles"][tp]
    use = (asu >= 0) & (gt >= a.gmin)
    hp = np.full(n, -1); hp[asu[use]] = np.flatnonzero(use)
    pi, pj = t["pi"], t["pj"]
    agree = use[pi] & (asu[pi] == pj)
    contra = ~agree & ((use[pi] & (asu[pi] != pj)) | ((hp[pj] >= 0) & (hp[pj] != pi)) | ((gt[pi] >= a.gmin) & (root[pj] == root[pi]) & (coord[pj] != coord[pi] + 1)))
    pr = (p2 + a.delta * agree - a.delta * contra).astype(np.float32)
    _, res = match(t, pr, 8, s)
    for k, (su, sc) in enumerate(res):
        OutSu[k, loc] = np.where(su >= 0, loc[np.maximum(su, 0)], -1); OutSc[k, loc] = sc
    info[s] = dict(merges=A.merges, coverage10=float(np.mean(gt >= 10)), asm_links=float(use.mean()), agree_pairs=int(agree.sum()), contra_pairs=int(contra.sum()),
                   changed_m0=float(np.mean(res[0][0] != stored)))
    log(f"sbj {s}: n {n} merges {A.merges} coverage>=10 {info[s]['coverage10']:.4f} asm links {info[s]['asm_links']:.4f} | changed member 0 vs stored {info[s]['changed_m0']:.4f}")
for k in range(8):
    OutSc[k] = qnorm(OutSu[k], OutSc[k], tsbj, REF)
    s_ = OutSu[k][OutSu[k] >= 0]; assert len(np.unique(s_)) == len(s_)
Z = np.load(os.path.join(HERE, a.oof))
np.savez(os.path.join(HERE, a.out), oof_succ=Z["oof_succ"], oof_score=Z["oof_score"], test_succ=OutSu, test_score=OutSc)
json.dump(dict(cfg=a.cfg, gmin=a.gmin, delta=a.delta, info={int(k): v for k, v in info.items()},
               changed_test_m0=float(np.mean(OutSu[0] != TSU[0]))), open(os.path.join(HERE, a.out[:-4] + "_info.json"), "w"), indent=1)
log(f"wrote {a.out}; test member-0 changed vs chain-rescored {np.mean(OutSu[0] != TSU[0]):.4f}")
