import numpy as np
K = r"E:\Claude code\wear\work\hanbat\keep"
m = np.load(K + r"\sim_meta.npz")
rec, start, y, sbj, fold, sensor = m["rec"], m["start"], m["y"], m["sbj"], m["fold"], m["sensor"]
t = start // 50
print("start%50 nonzero:", (start % 50 != 0).sum())
for r in np.unique(rec):
    ii = np.flatnonzero(rec == r)
    tt = np.sort(t[ii])
    d = np.diff(tt)
    # block order: first appearance of each label
    o = np.argsort(t[ii]); yy = y[ii][o]
    seq = []
    for v in yy:
        if v != 0 and (not seq or seq[-1] != v):
            seq.append(int(v))
    print(r, "sbj", np.unique(sbj[ii]), "fold", np.unique(fold[ii]), "n", len(ii), "tmin", tt[0], "tmax", tt[-1],
          "gaps>1:", (d > 1).sum(), "dup:", (d == 0).sum(), "null%%: %.2f" % (yy == 0).mean())
    print("   seq", seq)
