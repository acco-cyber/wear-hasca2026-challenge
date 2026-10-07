"""asmT: apply the asmB global offset solver (best refined K7 OOF of the two assembly implementations) to one fit.
  python run_asmT.py --fit K7|K9
OOF: permuted-node de-augmented simulation of the 22 training subjects with that fit's chain-rescored OOF votes
(testD/links_oof_<fit>_deaug.npz) and L3 candidates (<fit> keep4/link_logodds.npz); the one tuned threshold (wmin in
0.3/0.5/0.8) is chosen NESTED by subject fold (member-0 exact successor rate on the other folds). TEST: subjects 22-25
with testdeaug_chain_s*.npz and testD/links_test_<fit>_deaug.npz, wmin chosen on all 22 training subjects.
Fixed asmB parameters: thr 0.9, ratio 2, smin 0, l3w 0.3, l3k 3, chain_w 1, gmin 10, relink on, assembly score = 99th
percentile of qn_ref. Imports asmB's asmlib/solver read-only; writes only into this folder."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ASMB = os.path.join(os.path.dirname(HERE), "asmB")
sys.path.insert(0, ASMB)
from asmlib import load_sim, load_test, local_links, stage, fragments, TD, K7, K9   # noqa: E402
from solver import Solver, build_votes, pair_votes                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="K7", choices=["K7", "K9"]); ap.add_argument("--jobs", type=int, default=3)
ap.add_argument("--seed", type=int, default=12345)
a = ap.parse_args()
FIT = {"K7": K7, "K9": K9}[a.fit]
P = dict(thr=0.9, ratio=2.0, smin=0.0, l3w=0.3, l3k=3, chain_w=1.0, gmin=10)
WMINS = [0.3, 0.5, 0.8]
T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)

S = stage(FIT); sbj = S["oof_sbj"]; N = len(sbj); tsbj = S["test_sbj"]; NT = len(tsbj)
HIGH = float(np.percentile(np.load(os.path.join(FIT, "links.npz"))["qn_ref"], 99))
OOFL = np.load(os.path.join(TD, f"links_oof_{a.fit}_deaug.npz")); OSU, OSC = OOFL["oof_succ"], OOFL["oof_score"]
TSTL = np.load(os.path.join(TD, f"links_test_{a.fit}_deaug.npz")); TSU, TSC = TSTL["test_succ"], TSTL["test_score"]
assert (TSTL["oof_succ"] == OSU).all()
LZP = os.path.join(FIT, "link_logodds.npz")
BINS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 10 ** 9)]


def final_links(su, sc, asu, pv, high):
    """asmB final links: assembly successor in all members (score high) where found, else member k of the old links;
    old links into claimed tiles dropped and their sources relinked to the best free candidate; members one-to-one"""
    M, n = su.shape; has = asu >= 0; claimed = np.zeros(n, bool); claimed[asu[has]] = True
    nsu = su.copy(); nsc = sc.copy()
    pi, pj, pw = pv; o = np.argsort(-pw, kind="stable"); pi, pj = pi[o], pj[o]
    for k in range(M):
        x = su[k]; drop = (~has) & (x >= 0) & claimed[np.maximum(x, 0)]
        nsu[k] = np.where(has, asu, np.where(drop, -1, x)); nsc[k] = np.where(has, high, np.where(drop, -50.0, sc[k]))
        need = np.zeros(n, bool); need[drop] = True; used = np.zeros(n, bool); used[nsu[k][nsu[k] >= 0]] = True
        for i_, j_ in zip(pi[need[pi]], pj[need[pi]]):
            if need[i_] and not used[j_]:
                hit = su[:, i_] == j_
                nsu[k][i_] = j_; nsc[k][i_] = float(sc[hit, i_].max()) if hit.any() else -3.0
                need[i_] = False; used[j_] = True
        v = nsu[k][nsu[k] >= 0]; assert len(np.unique(v)) == len(v), k
        assert (nsu[k] != np.arange(n)).all()
    return nsu, nsc.astype(np.float32)


def frag_stats(d, thr):
    """label-free chain fragment statistics per limb (node-weighted mean fragment length, share of nodes in >=10)"""
    frag, off, flim, flen, fn = fragments(d["succ"], d["conf"], thr); out = {}
    for L in range(4):
        fl = flen[frag[L]]; out[L] = (float(fl.mean()), float(np.mean(fl >= 10)))
    return out


def solve_subject(d, su, sc, cand, Lo, wmins):
    sv0 = Solver(d, P["thr"])
    fa, fb, dd, w = build_votes(d, sv0.frag, sv0.off, P["thr"], su, sc, cand, Lo, P["l3w"], P["l3k"], P["chain_w"])
    pv = pair_votes(su, sc, cand, Lo, 0.3, 5)
    out = {}
    for wm in wmins:
        sv = Solver(d, P["thr"]); sv.solve(fa, fb, dd, w, wm, P["ratio"], P["smin"])
        asu, status, g, c, size = sv.derive(P["gmin"])
        nsu, nsc = final_links(su, sc, asu, pv, HIGH)
        out[wm] = dict(nsu=nsu, nsc=nsc, asu=asu, status=status, g=g, c=c, size=size, nm=sv.n_merge, nc=sv.n_coll)
    return out, frag_stats(d, P["thr"])


def run_sim(s):
    t0 = time.time()
    d = load_sim(s, a.seed); tr = d.pop("truth")          # truth kept apart: used only for scoring below
    su, sc = local_links(OSU, OSC, d["loc"], N)
    LZ = np.load(LZP); cand = LZ[f"oof_{s}_cand"].astype(np.int64); Lo = LZ[f"oof_{s}_L"].astype(np.float32)
    out, fs = solve_subject(d, su, sc, cand, Lo, WMINS)
    return dict(s=s, n=d["n"], loc=d["loc"], lim=d["lim"], anch=d["anch"], fold=d["fold"], su=su, sc=sc, out=out, fs=fs,
                tl=tr["tl"], y=tr["y"], tpos=tr["tpos"], t=time.time() - t0)


def run_test(s, wm):
    d = load_test(s, FIT)
    su, sc = local_links(TSU, TSC, d["loc"], NT)
    LZ = np.load(LZP); cand = LZ[f"test_{s}_cand"].astype(np.int64); Lo = LZ[f"test_{s}_L"].astype(np.float32)
    out, fs = solve_subject(d, su, sc, cand, Lo, [wm])
    return dict(s=s, n=d["n"], loc=d["loc"], lim=d["lim"], anch=d["anch"], su=su, sc=sc, out=out, fs=fs)


def lf_stats(rs, pick):
    """label-free statistics over a list of subject results; pick(r) -> chosen wmin of that subject"""
    cov, asm, ch0, chall, lk, lkb, agree, sizes_t, gsz, st, lim, anch = [], [], [], [], [], [], [], [], [], [], [], []
    for r in rs:
        o = r["out"][pick(r)]; ing = (o["g"] >= 0) & (o["size"] >= P["gmin"]); has = o["asu"] >= 0
        cov.append(ing); asm.append(has); ch0.append(o["nsu"][0] != r["su"][0]); chall.append((o["nsu"] != r["su"]).ravel())
        lk.append(o["nsu"][0] >= 0); lkb.append(r["su"][0] >= 0); agree.append(o["asu"][has] == r["su"][0][has])
        sizes_t.append(np.where(o["g"] >= 0, o["size"], 0)); st.append(o["status"]); lim.append(r["lim"]); anch.append(r["anch"])
        gg = o["g"][o["g"] >= 0]; _, cnt = np.unique(gg, return_counts=True); gsz.append(cnt)
    C = lambda v: np.concatenate(v)
    cov, asm, ch0, chall, lk, lkb, agree, sz, st, lim, anch, gsz = (C(v) for v in (cov, asm, ch0, chall, lk, lkb, agree, sizes_t, st, lim, anch, gsz))
    big = gsz[gsz >= P["gmin"]]
    dist = {"unanchored": float(np.mean(sz == 0))}
    for lo, hi in BINS:
        dist[f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"] = float(np.mean((sz >= lo) & (sz <= hi)))
    fsL = {L: [float(np.mean([r["fs"][L][0] for r in rs])), float(np.mean([r["fs"][L][1] for r in rs]))] for L in range(4)}
    return dict(n=int(len(cov)), coverage_ge10=float(cov.mean()), asm_link_share=float(asm.mean()),
                status_share={k: float(np.mean(st == k)) for k in range(5)},
                member0_changed=float(ch0.mean()), all_members_changed=float(chall.mean()),
                linked_new=float(lk.mean()), linked_base=float(lkb.mean()), asm_agree_base_member0=float(agree.mean()) if len(agree) else None,
                coverage_by_limb={L: float(cov[lim == L].mean()) for L in range(4)}, anchored_by_limb={L: float(anch[lim == L].mean()) for L in range(4)},
                tile_share_by_group_size=dist, n_groups_ge10=int(len(big)), groups_ge10_median=float(np.median(big)) if len(big) else 0.0,
                groups_ge10_p90=float(np.percentile(big, 90)) if len(big) else 0.0, groups_max=int(gsz.max()) if len(gsz) else 0,
                frag_len_thr09_by_limb=fsL)


def lab_stats(rs, pick):
    e_b, e_n, xb, xn, lkb, lkn, fo, ing, ae, ab = [], [], [], [], [], [], [], [], [], []
    for r in rs:
        o = r["out"][pick(r)]; tl, y = r["tl"], r["y"]; h = tl >= 0; b0, n0 = r["su"][0], o["nsu"][0]
        e_b.append(b0[h] == tl[h]); e_n.append(n0[h] == tl[h]); fo.append(np.full(h.sum(), r["fold"]))
        ing.append(((o["g"] >= 0) & (o["size"] >= P["gmin"]))[h])
        xb.append(np.where(b0 >= 0, (y != y[np.maximum(b0, 0)]) & (b0 != tl), False)); lkb.append(b0 >= 0)
        xn.append(np.where(n0 >= 0, (y != y[np.maximum(n0, 0)]) & (n0 != tl), False)); lkn.append(n0 >= 0)
        st1 = (o["asu"] >= 0) & h; ae.append(o["asu"][st1] == tl[st1]); ab.append(b0[st1] == tl[st1])
    C = lambda v: np.concatenate(v)
    e_b, e_n, xb, xn, lkb, lkn, fo, ing, ae, ab = (C(v) for v in (e_b, e_n, xb, xn, lkb, lkn, fo, ing, ae, ab))
    return dict(exact_base=float(e_b.mean()), exact_new=float(e_n.mean()),
                exact_in_groups_base=float(e_b[ing].mean()), exact_in_groups_new=float(e_n[ing].mean()),
                asm_link_exact=float(ae.mean()), base_exact_on_asm=float(ab.mean()),
                wrong_cross_label_base=float(xb.sum() / lkb.sum()), wrong_cross_label_new=float(xn.sum() / lkn.sum()),
                exact_by_fold_base=[float(e_b[fo == f].mean()) for f in range(5)], exact_by_fold_new=[float(e_n[fo == f].mean()) for f in range(5)])


# ------------------------------------------------------------------ simulation (OOF)
subs = [int(x) for x in np.unique(sbj)]
res = Parallel(n_jobs=a.jobs)(delayed(run_sim)(s) for s in sorted(subs, key=lambda s: -(sbj == s).sum()))
res = sorted(res, key=lambda r: r["s"])
log(f"fit {a.fit}: simulation of {len(res)} subjects done; HIGH {HIGH:.4f}")
# nested pick of wmin by fold (labels used only here, on the OTHER folds' subjects)
ts = S["true_succ"]; h = ts >= 0; fold = S["oof_fold"]
ex = {}
for wm in WMINS:
    Su = np.full((8, N), -1, np.int64)
    for r in res:
        nsu = r["out"][wm]["nsu"]; Su[:, r["loc"]] = np.where(nsu >= 0, r["loc"][np.maximum(nsu, 0)], -1)
    ex[wm] = Su[0] == ts
pick_fold = {}
for f in range(5):
    tr = h & (fold != f); sc_ = {wm: float(ex[wm][tr].mean()) for wm in WMINS}
    pick_fold[f] = max(WMINS, key=lambda k: sc_[k])
    log(f"fold {f}: other-fold exact {json.dumps({str(k): round(v, 4) for k, v in sc_.items()})} -> wmin {pick_fold[f]}")
full = {wm: float(ex[wm][h].mean()) for wm in WMINS}; wm_test = max(WMINS, key=lambda k: full[k])
log(f"all-subject exact {json.dumps({str(k): round(v, 4) for k, v in full.items()})} -> test wmin {wm_test}")
OSu = np.full((8, N), -1, np.int64); OSc = np.full((8, N), -50.0, np.float32)
for r in res:
    o = r["out"][pick_fold[r["fold"]]]; loc = r["loc"]
    OSu[:, loc] = np.where(o["nsu"] >= 0, loc[np.maximum(o["nsu"], 0)], -1); OSc[:, loc] = o["nsc"]
pk = lambda r: pick_fold[r["fold"]]
sim_lf = lf_stats(res, pk); sim_lab = lab_stats(res, pk)
log("SIM label-free " + json.dumps(sim_lf))
log("SIM scored     " + json.dumps(sim_lab))

# ------------------------------------------------------------------ test
tres = [run_test(s, wm_test) for s in (22, 23, 24, 25)]
TSu = TSU.astype(np.int64).copy(); TSc = TSC.astype(np.float32).copy(); per_sub = {}
for r in tres:
    o = r["out"][wm_test]; loc = r["loc"]
    TSu[:, loc] = np.where(o["nsu"] >= 0, loc[np.maximum(o["nsu"], 0)], -1); TSc[:, loc] = o["nsc"]
    per_sub[r["s"]] = lf_stats([r], lambda r_: wm_test); per_sub[r["s"]].update(merges=o["nm"], collisions=o["nc"])
    log(f"TEST sbj {r['s']}: n {r['n']} cov {per_sub[r['s']]['coverage_ge10']:.4f} asm {per_sub[r['s']]['asm_link_share']:.4f} "
        f"m0 changed {per_sub[r['s']]['member0_changed']:.4f} merges {o['nm']} coll {o['nc']}")
test_lf = lf_stats(tres, lambda r_: wm_test)
log("TEST label-free " + json.dumps(test_lf))
# per-subject simulation spread of the main label-free stats (for comparison with the 4 test subjects)
sim_sub = {r["s"]: {k: lf_stats([r], pk)[k] for k in ("coverage_ge10", "member0_changed", "asm_link_share")} for r in res}

# ------------------------------------------------------------------ checks + write
for nm, Su in (("oof", OSu), ("test", TSu)):
    for k in range(8):
        v = Su[k][Su[k] >= 0]; assert len(np.unique(v)) == len(v), (nm, k); assert (Su[k] != np.arange(Su.shape[1])).all()
assert TSu.shape == (8, NT) and OSu.shape == (8, N)
# each link stays inside its subject
assert (np.where(OSu >= 0, sbj[np.maximum(OSu, 0)] == sbj[None], True)).all()
assert (np.where(TSu >= 0, tsbj[np.maximum(TSu, 0)] == tsbj[None], True)).all()
outp = os.path.join(HERE, f"links_asm_{a.fit}.npz")
np.savez(outp, test_succ=TSu, test_score=TSc, oof_succ=OSu, oof_score=OSc)
info = dict(fit=a.fit, params=P, wmins=WMINS, nested_pick=pick_fold, test_wmin=wm_test, all_subject_exact=full, high=HIGH,
            sim_label_free=sim_lf, sim_scored=sim_lab, sim_per_subject=sim_sub, test_label_free=test_lf, test_per_subject=per_sub,
            test_member0_changed_vs_chain=float(np.mean(TSu[0] != TSU[0])), test_all_changed_vs_chain=float(np.mean(TSu != TSU)),
            oof_member0_changed_vs_chain=float(np.mean(OSu[0] != OSU[0])), oof_all_changed_vs_chain=float(np.mean(OSu != OSU)))
json.dump(info, open(os.path.join(HERE, f"info_{a.fit}.json"), "w"), indent=1)
log(f"wrote {outp}; test member0 changed {info['test_member0_changed_vs_chain']:.4f} (all members {info['test_all_changed_vs_chain']:.4f}); "
    f"oof member0 changed {info['oof_member0_changed_vs_chain']:.4f}")
