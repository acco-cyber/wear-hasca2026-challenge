"""Temporal-drift test: do video features of bouts close in time look more alike (beyond activity)?
Test: d(11,13)+d(12,14) < d(11,14)+d(12,13) (normal variants early, complex late); similar for other pairs."""
import pickle, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\transductive")
C = {}
for w in ("eval", "extra", "extra2"): C.update(pickle.load(open(rf"E:\Claude code\wear\exp\transductive\cache_{w}.pkl", "rb")))
print(list(C["sbj_0"].keys()), {k: getattr(v, "shape", None) for k, v in C["sbj_0"].items()})
def cosd(a, b): return 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b)
res = {}
for s, d in C.items():
    y = d["y"]; t = np.arange(len(y))
    for fk in ("v768", "vm", "vs"):
        X = np.asarray(d["F"][fk], np.float64); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        mu = {c: X[y == c].mean(0) for c in range(1, 19) if (y == c).sum() > 5}
        tm = {c: t[y == c].mean() for c in mu}
        def test(a, b, c, e):   # a,b same family early/late; c,e another family early/late
            if not all(k in mu for k in (a, b, c, e)): return None
            return cosd(mu[a], mu[c]) + cosd(mu[b], mu[e]) < cosd(mu[a], mu[e]) + cosd(mu[b], mu[c])
        r1 = test(11, 12, 13, 14)
        # generic: correlation over pairs of different-family activities between feature distance and |dt|
        FAM = np.zeros(19, int); FAM[1:6] = 1; FAM[6:11] = 2; FAM[11:13] = 3; FAM[13:15] = 4; FAM[15] = 5; FAM[16:18] = 6; FAM[18] = 7
        ks = list(mu); dd, dtt = [], []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if FAM[ks[i]] == FAM[ks[j]]: continue
                dd.append(cosd(mu[ks[i]], mu[ks[j]])); dtt.append(abs(tm[ks[i]] - tm[ks[j]]))
        from scipy.stats import spearmanr
        rho = spearmanr(dd, dtt).correlation
        res.setdefault(fk, []).append((r1, rho))
for fk, v in res.items():
    r1 = [a for a, b in v if a is not None]
    print(fk, "11/12 vs 13/14 pairing correct", sum(r1), "/", len(r1), "| mean spearman(dist, |dt|) over diff-family pairs", np.mean([b for a, b in v]).round(3))
