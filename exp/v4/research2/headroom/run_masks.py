"""Lever 4/5 at decode level (K7, OOF only, learned nested counts):
  session : ORACLE session membership = every recording split at ONE time point into two contiguous halves (the split
            that best separates the two 9-exercise protocol blocks); a tile may only take null or a class that truly
            occurs in its half. Mask on the stage-B input (log 1e-6 for disallowed) and on every graph output.
  null    : ORACLE null/activity mask (true null tiles -> {null}, activity tiles -> {1..18}) as the same hard constraint.
  null3   : the null/activity mask only on tiles within 3 tiles of a true bout edge (in true time order).
Also writes the post-hoc version of the 2-half session oracle on the fused b4wa arrays."""
import sys
import numpy as np
from hlib import *

B1 = [1, 2, 3, 6, 7, 11, 12, 13, 14]; BLK = np.zeros(19, int); BLK[B1] = 1; BLK[[c for c in range(1, 19) if c not in B1]] = 2
D = setup(); y, sbj, fold = D["y"], D["sbj"], D["fold"]; n = len(y)
bid, dist = bouts(D)


def half_sessions():
    ses = np.full(n, -1); M = np.zeros((n, N_CLS)); sid = 0; pur = []
    for ii in D["order"]:
        b = BLK[y[ii]]; a1 = np.r_[0, np.cumsum(b == 1)]; a2 = np.r_[0, np.cumsum(b == 2)]
        # split at t: first part [0, t), second [t, n). score = block-1 tiles before + block-2 tiles after (or reverse)
        s12 = a1 + (a2[-1] - a2); s21 = a2 + (a1[-1] - a1)
        t = int(np.argmax(s12)) if s12.max() >= s21.max() else int(np.argmax(s21))
        pur.append(max(s12.max(), s21.max()) / max(1, (b > 0).sum()))
        for part in (ii[:t], ii[t:]):
            if len(part) == 0:
                continue
            ses[part] = sid; cl = np.unique(np.r_[0, y[part]]); M[np.ix_(part, cl)] = 1; sid += 1
    log(f"2-half sessions: block purity per recording {np.round(pur, 3).tolist()}; allowed classes per session "
        f"{[int(M[ses == s_][0].sum() - 1) for s_ in range(sid)]}")
    return ses, M


ses, MS = half_sessions()
MN = np.zeros((n, N_CLS)); MN[y == 0, 0] = 1; MN[y > 0, 1:] = 1
MN3 = np.ones((n, N_CLS)); near = dist <= 3; MN3[near] = MN[near]
np.savez_compressed(os.path.join(HERE, "masks.npz"), ses=ses, MS=MS.astype(np.int8), MN=MN.astype(np.int8), MN3=MN3.astype(np.int8))

# post-hoc on fused b4wa
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64); labo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64)
for nm, M in (("session(2-half)", MS), ("null", MN), ("null3", MN3)):
    Qm = Qo * M; Qm /= Qm.sum(1, keepdims=True); la = Qm.argmax(1); fx = labo.copy(); bad = M[np.arange(n), labo] == 0; fx[bad] = la[bad]
    log(f"POSTHOC b4wa {nm}: refined labels violating {bad.sum()}, fixed -> {macro_f1(y, fx):.4f} ({macro_f1(y, fx) - macro_f1(y, labo):+.4f}); "
        f"masked Q argmax {macro_f1(y, la):.4f} vs {macro_f1(y, Qo.argmax(1)):.4f}")

which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["session", "null", "null3"]
res = {}
for nm in which:
    M = {"session": MS, "null": MN, "null3": MN3}[nm]
    logp = H.lsm(D["Bo"].astype(np.float64) + np.log(M + 1e-6)).astype(np.float32)
    r = decode(D, D["Lo"], "learned", logp=logp, mask=M)
    ref = refine_oof(D, r["fin"], D["Lo"], r["P"], r["Q"], Bo=logp)
    bad = M[np.arange(n), ref] == 0; ref2 = ref.copy(); ref2[bad] = r["fin"][bad]
    res[nm] = (macro_f1(y, r["fin"]), macro_f1(y, ref), macro_f1(y, ref2))
    np.savez_compressed(os.path.join(HERE, f"k7_mask_{nm}.npz"), fin=r["fin"], ref=ref, ref2=ref2)
for k, v in res.items():
    log(f"RESULT mask-decode {k:8s} pre-refiner {v[0]:.4f} refined {v[1]:.4f} refined+mask {v[2]:.4f}")
