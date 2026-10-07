"""Post-hoc oracles on decoded labels (no re-decode): error anatomy, null/activity mask oracle, session(block) oracle.
Run on the fused b4wa arrays (OOF 0.9336) and on K7 (kernel refined labels)."""
import sys
import numpy as np
from collections import Counter
from hlib import *

B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; B2 = [4, 5, 8, 9, 10, 15, 16, 17, 18]
BLK = np.zeros(19, int); BLK[B1] = 1; BLK[B2] = 2

D = setup(); y, fold, sbj = D["y"], D["fold"], D["sbj"]; n = len(y)
bid, dist = bouts(D)


def f1(p):
    return macro_f1(y, p)


def pf(p):
    return " ".join(f"{macro_f1(y[fold == f_], p[fold == f_]):.4f}" for f_ in range(FOLDS))


def sinkhorn_keep(Qm, Qref):
    """re-calibrate a masked Q to the per-subject column sums of the unmasked Q (keeps the decode's counts)"""
    out = np.empty_like(Qm)
    for s in np.unique(sbj):
        ii = sbj == s; t = Qref[ii].sum(0); Q = np.clip(Qm[ii], 1e-12, None).copy()
        Q *= (Qm[ii] > 0)
        for _ in range(50):
            Q *= (t / np.maximum(Q.sum(0), 1e-9))[None]; Q /= np.maximum(Q.sum(1, keepdims=True), 1e-300)
        out[ii] = Q
    return out


# ---- session structure in the training recordings: block sequence along true time
sw = []
for ii in D["order"]:
    lab = y[ii]; a = lab[lab > 0]; b = BLK[a]; chg = np.flatnonzero(b[1:] != b[:-1])
    sw.append(len(chg))
log(f"block switches per recording along time (activity tiles): {sw}")


def true_session(D):
    """oracle session per tile: runs of one protocol block along true time (activity tiles), null tiles take the session
    of the nearest activity tile; also returns the true class set per session"""
    ses = np.full(n, -1); sid = 0; classes = {}
    for ii in D["order"]:
        lab = y[ii]; a = np.flatnonzero(lab > 0); b = BLK[lab[a]]
        run = np.r_[0, np.cumsum(b[1:] != b[:-1])]          # run id per activity tile
        # absorb tiny runs (< 30 tiles: isolated relabel noise) into the neighbouring run
        t = np.arange(len(ii)); pos = np.searchsorted(a, t); lo = a[np.clip(pos - 1, 0, len(a) - 1)]; hi = a[np.clip(pos, 0, len(a) - 1)]
        near = np.where(np.abs(t - lo) <= np.abs(hi - t), lo, hi)
        rid = np.zeros(len(ii), int); rid[a] = run; rid = rid[near]
        ses[ii] = rid + sid
        for r_ in np.unique(rid):
            classes[r_ + sid] = set(np.unique(lab[rid == r_]).tolist()) | {0}
        sid += rid.max() + 1
    return ses, classes


ses, classes = true_session(D)
log(f"oracle sessions: {len(classes)} over {len(D['order'])} recordings; classes per session "
    f"{sorted(Counter(len(c) - 1 for c in classes.values()).items())}")
MASK = np.zeros((n, N_CLS))
for s_, cl in classes.items():
    MASK[np.ix_(ses == s_, sorted(cl))] = 1


def anatomy(name, lab, Q):
    err = lab != y; log(f"=== {name}: F1 {f1(lab):.4f} | errors {err.sum()} ({err.mean():.4f}) | per fold {pf(lab)}")
    cats = {"null->act": (y == 0) & (lab > 0), "act->null": (y > 0) & (lab == 0), "act->act'": (y > 0) & (lab > 0) & (lab != y)}
    for c, m in cats.items():
        fx = lab.copy(); fx[m] = y[m]
        near = [np.sum(m & (dist <= d)) for d in (1, 3, 5, 10)]
        log(f"  {c:9s}: {m.sum():5d} tiles (dist<=1/3/5/10 from a true bout edge: {near}); fix all -> F1 {f1(fx):.4f} ({f1(fx) - f1(lab):+.4f})")
    for d in (1, 2, 3, 5, 10):
        m = err & (dist <= d); fx = lab.copy(); fx[m] = y[m]
        log(f"  all errors within {d:2d} tiles of a true edge: {m.sum():5d} -> F1 {f1(fx):.4f} ({f1(fx) - f1(lab):+.4f})")
    # bout-level: majority of decoded labels inside each true bout
    maj = np.zeros(bid.max() + 1, int); good = np.zeros(bid.max() + 1, bool)
    for b in np.unique(bid):
        ii = np.flatnonzero(bid == b); c = np.bincount(lab[ii], minlength=N_CLS); maj[b] = c.argmax()
    wrongb = maj != np.array([y[np.flatnonzero(bid == b)[0]] for b in range(bid.max() + 1)])
    act_b = np.array([y[np.flatnonzero(bid == b)[0]] > 0 for b in range(bid.max() + 1)])
    lens = np.bincount(bid)
    log(f"  true bouts {len(maj)} ({act_b.sum()} activity); activity bouts with wrong majority {np.sum(wrongb & act_b)} "
        f"(tiles {lens[wrongb & act_b].sum()}); null bouts wrong majority {np.sum(wrongb & ~act_b)} (tiles {lens[wrongb & ~act_b].sum()})")
    fx = lab.copy(); m = wrongb[bid] & (y > 0); fx[m] = y[m]
    log(f"  fix whole-bout activity errors only -> {f1(fx):.4f} ({f1(fx) - f1(lab):+.4f})")
    bm = maj[bid]; log(f"  bout-majority oracle (every tile gets its true bout's decoded majority) -> {f1(bm):.4f}")
    # null-mask oracles
    Qa = Q.copy(); Qa[:, 0] = 0; alt = Qa.argmax(1)
    for d in (1, 2, 3, 5, 10, 10 ** 7):
        m = dist <= d; fx = lab.copy()
        fx[m & (y == 0)] = 0; mm = m & (y > 0) & (lab == 0); fx[mm] = alt[mm]
        log(f"  NULL-MASK oracle on tiles within {d if d < 10 ** 7 else 'any':>3} of a true edge -> F1 {f1(fx):.4f} ({f1(fx) - f1(lab):+.4f})")
    # session oracle: mask disallowed classes
    Qm = Q * MASK; Qm /= Qm.sum(1, keepdims=True); la = Qm.argmax(1)
    fx = lab.copy(); bad = MASK[np.arange(n), lab] == 0; fx[bad] = la[bad]
    log(f"  SESSION oracle: labels in a wrong session {bad.sum()}; replace by masked-Q argmax -> {f1(fx):.4f} ({f1(fx) - f1(lab):+.4f}); "
        f"masked Q argmax alone {f1(la):.4f} (unmasked Q argmax {f1(Q.argmax(1)):.4f}); masked + re-Sinkhorn {f1(sinkhorn_keep(Qm, Q).argmax(1)):.4f}")
    # top confusions
    log("  top confusions (true,pred): " + str(Counter(zip(y[err].tolist(), lab[err].tolist())).most_common(10)))
    # per-class F1
    from sklearn.metrics import f1_score
    pc = f1_score(y, lab, average=None, labels=list(range(N_CLS)))
    log("  per-class F1: " + " ".join(f"{c}:{v:.3f}" for c, v in enumerate(pc)))
    # per-subject F1
    log("  per-subject F1: " + " ".join(f"{s}:{macro_f1(y[sbj == s], lab[sbj == s]):.3f}" for s in np.unique(sbj)))


Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64); labo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64)
assert len(Qo) == n
log(f"b4wa: pre-refiner (Q argmax) {f1(Qo.argmax(1)):.4f}, refined {f1(labo):.4f}")
anatomy("b4wa fused refined", labo, Qo)
anatomy("b4wa fused pre-refiner", Qo.argmax(1), Qo)
anatomy("K7 kernel refined", D["F"]["ref_oof"].astype(np.int64), D["F"]["QB_OOF"].astype(np.float64))
np.savez_compressed(os.path.join(HERE, "sessions.npz"), ses=ses, mask=MASK.astype(np.int8), bid=bid, dist=dist)
