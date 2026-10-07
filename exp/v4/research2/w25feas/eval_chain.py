"""summarise per-limb chain caches: exact successor overall / by limb / by activity / dynamic vs static, same-label
successor rate, rank-1 rate, mean correct-run length.  python eval_chain.py nll3 [subjects]"""
import os, sys, glob
import numpy as np
from common import *
kind = sys.argv[1]
S = load_stage(); y = S["oof_y"]
NAMES = ["null", "jogging", "jog-rot", "jog-skip", "jog-side", "jog-butt", "str-tric", "str-lunge", "str-shoul", "str-hams", "str-lumb",
         "push-ups", "push-ups-c", "sit-ups", "sit-ups-c", "burpees", "lunges", "lunges-c", "bench-dips"]
files = sorted(glob.glob(os.path.join(OUT, "cache", f"chain_{kind}_s*.npz")))
if len(sys.argv) > 2:
    keep = set(int(x) for x in sys.argv[2].split(",")); files = [f for f in files if int(f.split("_s")[-1][:-4]) in keep]
ex = {L: [] for L in range(4)}; sl = {L: [] for L in range(4)}; r1 = {L: [] for L in range(4)}; lab = []; dyn = {L: [] for L in range(4)}; runs = {L: [] for L in range(4)}
subj = []
for f in files:
    z = np.load(f); ii = z["rows"]; tl = z["true_loc"]; h = tl >= 0
    s = int(f.split("_s")[-1][:-4]); ii_ = ii
    lab.append(y[ii][h]); subj.append(np.full(h.sum(), s))
    for L in range(4):
        su = z[f"succ{L}"]; ok = su[h] == tl[h]; ex[L].append(ok)
        sl[L].append(np.where(su[h] >= 0, y[ii][np.maximum(su[h], 0)] == y[ii][tl[h]], False))
        r1[L].append(z[f"rank{L}"][h] == 0 if f"rank{L}" in z.files else np.full(h.sum(), np.nan))
        # correct-run lengths along the true order
        o = np.r_[0, np.flatnonzero(~ok) + 1, len(ok) + 1]; runs[L].append(np.diff(o))
lab = np.concatenate(lab); subj = np.concatenate(subj)
print(f"cost {kind}: {len(files)} subjects, {len(lab)} true links per limb")
LN = ["right_arm", "right_leg", "left_leg", "left_arm"]
allx = []
for L in range(4):
    e = np.concatenate(ex[L]); allx.append(e); s_ = np.concatenate(sl[L]); r = np.concatenate(r1[L]); rl = np.concatenate(runs[L])
    print(f"  {LN[L]:9s}: exact {e.mean():.3f} | same-label succ {s_.mean():.3f} | true succ ranked 1st {r.mean():.3f} | mean correct run {np.average(rl, weights=rl):.1f} tiles (tile-weighted)")
E = np.stack(allx, 1)
print(f"  all limbs: exact {E.mean():.3f}; >=1 of 4 limbs exact {E.any(1).mean():.3f}; all 4 exact {E.all(1).mean():.3f}")
print("  by activity (exact, mean over limbs | arms | legs | share):")
for c in range(19):
    m = lab == c
    if m.sum():
        print(f"    {NAMES[c]:11s} {E[m].mean():.3f} | {E[m][:, [0, 3]].mean():.3f} | {E[m][:, [1, 2]].mean():.3f} | {m.mean():.3f}")
print("  by subject:", " ".join(f"{s}:{E[subj == s].mean():.3f}" for s in np.unique(subj)))
