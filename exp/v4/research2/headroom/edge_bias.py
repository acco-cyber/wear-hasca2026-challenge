"""Null-edge error profile of the fused b4wa labels in true time order: error rate by offset from every true
null->activity start and activity->null end (negative offset = tile before the edge). A systematic shift would be a
label-free correctable bias; a symmetric bump at offsets -1/+1 is annotation-level edge ambiguity."""
import numpy as np
from hlib import *

D = setup(); y = D["y"]
lab = np.load(os.path.join(SUBS, "sub_v4c_b4wa_labo.npy")).astype(np.int64)
R = 6
st = {o: [0, 0, 0] for o in range(-R, R)}; en = {o: [0, 0, 0] for o in range(-R, R)}   # [tiles, errors, null-errors]
for ii in D["order"]:
    yy, ll = y[ii], lab[ii]; n = len(ii)
    for t in range(1, n):
        if yy[t - 1] == 0 and yy[t] > 0:   # activity starts at t
            tab = st
        elif yy[t - 1] > 0 and yy[t] == 0:  # activity ends at t-1 (null starts at t)
            tab = en
        else:
            continue
        for o in range(-R, R):
            g = t + o
            if 0 <= g < n:
                tab[o][0] += 1; e = ll[g] != yy[g]; tab[o][1] += e; tab[o][2] += e and (ll[g] == 0 or yy[g] == 0)
for nm, tab in (("activity START (offset 0 = first activity tile)", st), ("activity END (offset 0 = first null tile after)", en)):
    log(nm)
    log("  offset: " + " ".join(f"{o:>6d}" for o in range(-R, R)))
    log("  err %:  " + " ".join(f"{100 * tab[o][1] / max(tab[o][0], 1):6.1f}" for o in range(-R, R)))
    log("  null%:  " + " ".join(f"{100 * tab[o][2] / max(tab[o][0], 1):6.1f}" for o in range(-R, R)))
# signed bias: decoded activity extent vs true, per activity bout (first/last decoded tile == true class within +-10)
d_start, d_end = [], []
for ii in D["order"]:
    yy, ll = y[ii], lab[ii]; n = len(ii)
    chg = np.flatnonzero(np.r_[True, yy[1:] != yy[:-1]]); ends = np.r_[chg[1:], n]
    for a, b in zip(chg, ends):
        c = yy[a]
        if c == 0 or b - a < 20 or a < 10 or b > n - 10:
            continue
        if yy[a - 1] != 0 or yy[b] != 0:
            continue
        w = ll[a - 10:b + 10] == c; idx = np.flatnonzero(w)
        if len(idx) == 0:
            continue
        d_start.append(idx[0] - 10); d_end.append((idx[-1] + a - 10) - (b - 1))
d_start, d_end = np.array(d_start), np.array(d_end)
log(f"bouts {len(d_start)}: decoded start - true start: mean {d_start.mean():+.2f} median {np.median(d_start):+.0f} | "
    f"hist -3..3 {[int(np.sum(d_start == k)) for k in range(-3, 4)]}")
log(f"               decoded end - true end:     mean {d_end.mean():+.2f} median {np.median(d_end):+.0f} | "
    f"hist -3..3 {[int(np.sum(d_end == k)) for k in range(-3, 4)]}")
