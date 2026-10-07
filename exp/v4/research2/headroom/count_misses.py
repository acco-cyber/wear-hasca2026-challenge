"""The largest count misses of the fused decode on test-like subjects: what goes wrong there"""
import numpy as np
from hlib import *

D = setup(); y, sbj = D["y"], D["sbj"]
Qo = np.load(os.path.join(SUBS, "sub_v4c_b4wa_Qo.npy")).astype(np.float64); lab = Qo.argmax(1)
tg_f = {int(s): Qo[sbj == s].sum(0) for s in np.unique(sbj)}; TT = true_counts(D)
rows = []
for s in tg_f:
    if s in (0, 2, 14):
        continue
    for c in range(1, N_CLS):
        e = tg_f[s][c] - TT[s][c]
        if abs(e) > 15:
            ii = sbj == s
            got = np.bincount(y[ii & (lab == c)], minlength=N_CLS)          # true labels of tiles decoded as c
            went = np.bincount(lab[ii & (y == c)], minlength=N_CLS)         # decoded labels of true-c tiles
            rows.append((s, c, round(float(tg_f[s][c]), 0), TT[s][c], {k: int(v) for k, v in enumerate(got) if v and k != c},
                         {k: int(v) for k, v in enumerate(went) if v and k != c}))
for r in sorted(rows, key=lambda r: -abs(r[2] - r[3])):
    log(f"sbj {r[0]:2d} class {r[1]:2d}: target {r[2]:.0f} true {r[3]:.0f} | decoded-as-{r[1]} true labels {r[4]} | true-{r[1]} decoded as {r[5]}")
