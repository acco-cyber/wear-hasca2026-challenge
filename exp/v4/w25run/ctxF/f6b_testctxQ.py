"""TEST ctxQ: from each of our tiles walk the test chain of its limb; reached rows that are twins of our tiles of the same limb
carry that tile's kernel output QB_TEST (per fit).  -> cache/ctxQ_test_<fit>.npz (W{W}, n{W}); nested choice of (W, v)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
from f3b_ctxB import ctxB

T0 = time.time()
tz = np.load(os.path.join(CACHE, "test_twins.npz")); twin = tz["twin"]
CH = {s: np.load(os.path.join(CACHE, f"test_chain_s{s}.npz")) for s in (22, 23, 24, 25)}
for fit, d in (("K7", K7), ("K9", K9)):
    st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True); ids = st["ids"]
    sens = st["sensor_test"].astype(np.int64); tsb = st["test_sbj"].astype(np.int64)
    Q = lsm(np.log(np.clip(st["QB_TEST"].astype(np.float64), 1e-6, None))); out = {}
    for Wn in (4, 8, 16):
        acc = np.zeros((len(ids), N_CLS)); wsum = np.zeros(len(ids))
        for s in (22, 23, 24, 25):
            z = CH[s]
            for L in range(4):
                rows = z[f"rows{L}"]; n = len(rows); pos_of = np.full(48936, -1); pos_of[rows] = np.arange(n)
                mine = np.flatnonzero((tsb == s) & (sens == L)); start = pos_of[twin[mine]]; assert (start >= 0).all()
                mark = np.zeros(n, bool); mark[start] = True; Qr = np.zeros((n, N_CLS)); Qr[start] = Q[mine]
                a_, w_ = ctxB(Qr, mark, z[f"succ{L}"].astype(np.int64), z[f"conf{L}"].astype(np.float32), start, Wn)
                acc[mine] = a_; wsum[mine] = w_
        out[f"W{Wn}"] = acc.astype(np.float32); out[f"n{Wn}"] = wsum.astype(np.float32)
        o = np.load(os.path.join(CACHE, f"ctxQ_oof_{fit}.npz"))[f"n{Wn}"]
        print(f"{fit} test ctxQ W={Wn}: mean neighbour weight {wsum.mean():.3f} (OOF {o.mean():.3f}); argmax change of B2 at v=0.3: "
              f"test {np.mean(lsm(st['B2_TEST'].astype(np.float64) + 0.3 * acc).argmax(1) != st['B2_TEST'].argmax(1)):.4f}", flush=True)
    np.savez(os.path.join(CACHE, f"ctxQ_test_{fit}.npz"), ids=ids, **out)
# nested choice of (W, v) on OOF tile F1
for fit, d in (("K7", K7), ("K9", K9)):
    st = np.load(os.path.join(d, "stage.npz"), allow_pickle=True); y = st["oof_y"].astype(np.int64); fold = st["oof_fold"].astype(np.int64)
    B2 = st["B2_OOF"].astype(np.float64); cq = np.load(os.path.join(CACHE, f"ctxQ_oof_{fit}.npz"))
    res = {(Wn, v): lsm(B2 + v * cq[f"W{Wn}"].astype(np.float64)).argmax(1) for Wn in (4, 8, 16) for v in (0.1, 0.2, 0.3, 0.5, 0.8)}
    tab = {k: [macro_f1(y[fold != f], p[fold != f]) for f in range(5)] for k, p in res.items()}
    nest = np.zeros(len(y), np.int64); ch = []
    for f in range(5):
        b = max(tab, key=lambda k: tab[k][f]); ch.append(b); nest[fold == f] = res[b][fold == f]
    print(f"{fit} NESTED ctxQ choice {ch} -> tile F1 {macro_f1(y, nest):.4f} vs B2 {macro_f1(y, B2.argmax(1)):.4f}")
print(f"[{time.time() - T0:.0f}s]")
