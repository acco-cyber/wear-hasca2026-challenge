import json, sys
G = {}
g = []
for k in (5, 10):
    for beta in (0.3, 0.6, 1.0):
        g.append(dict(clf="none", spec="v768", aug_k=k, aug_beta=beta))
for k in (5, 10):
    for al in (0.5, 0.8):
        for w in (0.5, 1.0):
            g.append(dict(clf="ksmooth", spec="v768", k=k, alpha=al, w=w, rounds=1))
for k in (5, 10):
    g.append(dict(clf="knn", spec="v768", k=k, alpha=0.9, w=0.5, sel="marg", q=0.5, rounds=2))
G["g2"] = g
g = []
# per-subject recalibration on base log-probs
g.append(dict(clf="lr", spec="lp", C=1.0, sel="marg", q=0.5, w=0.5, rounds=1))
g.append(dict(clf="lr", spec="lp", C=1.0, sel="agree", w=1.0, rounds=1))
g.append(dict(clf="lr", spec="lp+v768", C=0.1, sel="marg", q=0.5, w=0.5, rounds=1))
# stronger feature-space smoothing
g.append(dict(clf="ksmooth", spec="v768", k=10, alpha=0.8, w=1.5, rounds=1))
g.append(dict(clf="ksmooth", spec="v768", k=10, alpha=0.9, w=1.0, rounds=1))
g.append(dict(clf="ksmooth", spec="v768", k=20, alpha=0.8, w=1.0, rounds=1))
g.append(dict(clf="ksmooth", spec="vm", k=10, alpha=0.8, w=1.0, rounds=1))
# label spreading of pseudo-labels
for k, al, w, sel, q in ((5, 0.9, 1.0, "marg", 0.5), (10, 0.9, 0.5, "marg", 0.5), (5, 0.99, 0.5, "marg", 0.5), (5, 0.9, 0.5, "marg", 0.7),
                         (5, 0.9, 0.5, "agree", 0.5), (5, 0.9, 0.5, "all", 0.5), (5, 0.9, 0.3, "marg", 0.5)):
    g.append(dict(clf="knn", spec="v768", k=k, alpha=al, w=w, sel=sel, q=q, rounds=1))
G["g3"] = g
KS = dict(clf="ksmooth", spec="v768", k=20, alpha=0.8, w=1.0)
def KN(sel, w=0.5, k=5, al=0.9, q=0.7): return dict(clf="knn", spec="v768", k=k, alpha=al, w=w, sel=sel, q=q)
g = [dict(KS, rounds=1)]
for sel in ("marg", "agree", "all"): g.append(dict(KN(sel), rounds=1))
g.append(dict(KN("all", w=1.0), rounds=1))
g.append(dict(KN("agree", w=0.75), rounds=1))
g.append(dict(KN("agree", k=10), rounds=1))
g.append(dict(steps=[KS, KN("agree")]))
g.append(dict(steps=[KS, KN("all")]))
g.append(dict(steps=[KN("agree"), KN("agree")]))
g.append(dict(steps=[KN("all"), KN("all")]))
G["g4"] = g
A = KN("all", w=1.0)
g = [dict(A, w=1.5), dict(A, w=2.0), dict(A, k=3), dict(A, k=8), dict(A, k=12), dict(A, alpha=0.8), dict(A, alpha=0.95),
     dict(A, d_vid=32), dict(A, d_vid=128), dict(A, eps=0.05), dict(A, eps=0.2), dict(steps=[A, A]), dict(A, spec="vm+vs+vd")]
G["g5"] = g
A2 = dict(A, w=2.0)
g = [dict(A, w=3.0), dict(A, w=4.0), dict(A, w=6.0), dict(A2, alpha=0.95), dict(A2, d_vid=128), dict(A2, eps=0.05), dict(A2, eps=0.2),
     dict(steps=[A2, A]), dict(steps=[A2, A2]), dict(A2, alpha=0.95, d_vid=128)]
G["g6"] = g
A6 = dict(A, w=6.0)
g = [dict(A, w=10.0), dict(A, w=20.0), dict(steps=[A6, A6]), dict(steps=[A6, A6, A6]), dict(A6, alpha=0.95), dict(A6, k=8), dict(A6, k=3),
     dict(A6, d_vid=128), dict(A6, alpha=0.8), dict(A6, spec="vm+vs+vd")]
G["g7"] = g
for name, g in G.items():
    json.dump(g, open(f"{name}.json", "w"), indent=0); print(name, len(g))
