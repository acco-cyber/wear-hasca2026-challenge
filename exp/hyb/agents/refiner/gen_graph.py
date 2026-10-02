"""Run the graph_lab recipe (extra links = our chain links, prior 0.3, counts 0.3, NO gate) and keep every array the
refiner needs.  python gen_graph.py cv|test   -> graph_<mode>.npz in this folder"""
import os, sys, pickle, time
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, pandas as pd
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
import graph_lab as G
from hanbat_stack import KEEP, TEST_CFGS, TRAIN_SETS, N_CLS, macro_f1
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb"); OUT = os.path.dirname(os.path.abspath(__file__))
PRIOR, COUNTS = 0.3, 0.3
mode = sys.argv[1]
t0 = time.time()
if mode == "cv":
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    l0 = np.load(os.path.join(KEEP, "links_L0.npz"))
    dd = dict(logp=np.load(os.path.join(W, "work", "hanbat", "cv_base_oof_logp_b.npy")).astype(np.float32),
              emb=np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32),
              grp=sm["sbj"], sbj=sm["sbj"], succ=l0["oof_succ"].astype(np.int64), score=l0["oof_score"].astype(np.float32), sets=TRAIN_SETS)
    rows = np.load(os.path.join(HYB, "rows.npz")); o2t = rows["ours_to_theirs"]
    R = pickle.load(open(os.path.join(W, "exp", "pl", "labels_v3bv1f.pkl"), "rb")); ours = np.full(len(sm["y"]), -1, np.int64)
    for s, d in R.items():
        ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
    has = ours >= 0
    sys.path.insert(0, os.path.join(W, "exp", "transductive")); from tlib import load_structs
    s2 = np.full(len(has), -1, np.int64); c2 = np.full(len(has), -50.0, np.float32)
    for which in ("eval", "extra", "extra2"):
        for s, st in load_structs(which).items():
            a0, n = int(st["a"]), int(st["n"]); su = np.asarray(st["succ0"]); sc = np.asarray(st["sc"], np.float32)
            ok = (su >= 0) & (sc >= -6.0); rows_t = o2t[a0 + np.arange(n)]
            s2[rows_t[ok]] = o2t[a0 + su[ok]]; c2[rows_t[ok]] = sc[ok]
    ours0 = np.where(has, ours, 0)
else:
    bl = np.load(os.path.join(KEEP, "blend.npz")); l2 = np.load(os.path.join(KEEP, "links_L2_test.npz"))
    sbj = bl["test_sbj"].astype(np.int64)
    dd = dict(logp=np.load(os.path.join(KEEP, "test_logp_b.npy")).astype(np.float32), emb=np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32),
              grp=sbj, sbj=sbj, succ=l2["succ"].astype(np.int64), score=l2["score_qn"].astype(np.float32), sets={})
    ours = pd.read_csv(os.path.join(W, "subs", "sub_e44_vote9.csv")).sort_values("id").target_feature.to_numpy().astype(np.int64)
    has = np.ones(len(ours), bool)
    struct = pickle.load(open(os.path.join(W, "work", "test_structure.pkl"), "rb"))
    s2 = np.full(len(ours), -1, np.int64); c2 = np.full(len(ours), -50.0, np.float32)
    for s, st in struct.items():
        idx = np.asarray(st["idx"]); su = np.asarray(st["succ0"]); sc = np.asarray(st["sc"], np.float32); ok = (su >= 0) & (sc >= -6.0)
        s2[idx[ok]] = idx[su[ok]]; c2[idx[ok]] = sc[ok]
    ours0 = ours
dd.update(succ2=s2, score2=c2, xl_w=1.0, xl_b=-2.0, xlinks=[])
sbj = dd["sbj"]; sets = dd["sets"]
cover = {int(s): bool(has[sbj == s].all()) for s in np.unique(sbj)}
targets = G.make_targets(sbj, sets, ours0, COUNTS, cover)
OH = np.full((len(sbj), N_CLS), 1.0 / N_CLS); eps = 0.1
OH[has] = eps / N_CLS; OH[np.flatnonzero(has), ours0[has]] += 1 - eps
P = G.graph_P2(dd, TEST_CFGS, targets, OH, PRIOR)
base, Q = G.finish2(P, sbj, targets)
gl, g = G.apply_gate(base, Q, ours0, has, "0.55", "top2")
T = np.stack([targets[int(s)] for s in np.unique(sbj)])
np.savez_compressed(os.path.join(OUT, f"graph_{mode}.npz"), P=P.astype(np.float32), Q=Q.astype(np.float32), base=base, ours=ours0, has=has,
                    s2=s2, c2=c2, succ=dd["succ"], score=dd["score"], sbj=sbj, gate_lab=gl, gate=g, T=T, T_sbj=np.unique(sbj))
print(f"{mode}: done in {time.time() - t0:.0f}s; gated {g.mean():.4f}", flush=True)
if mode == "cv":
    y = sm["y"]; S = has
    print(f"F1(S) base {macro_f1(y[S], base[S]):.4f} gate {macro_f1(y[S], gl[S]):.4f}; F1(all) base {macro_f1(y, base):.4f} gate {macro_f1(y, gl):.4f}")
else:
    ref = pd.read_csv(os.path.join(W, "subs", "sub_gl7_xl_p03c03_top2_055.csv")).sort_values("id").target_feature.to_numpy()
    print(f"agreement of re-run gate labels with sub_gl7_xl_p03c03_top2_055: {np.mean(gl == ref):.5f}; base vs ref {np.mean(base == ref):.4f}")
