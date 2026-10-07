import numpy as np, sys
from asmlib import *
S = stage(); subs = [int(x) for x in np.unique(S["oof_sbj"])]
bins = [0, .3, .5, .7, .8, .9, .95, 1.01]
acc = np.zeros((4, len(bins) - 1)); cnt = np.zeros((4, len(bins) - 1))
fr = {t: [] for t in (0.5, 0.7, 0.8, 0.9)}
pp1 = []; anr = np.zeros(4); anc = np.zeros(4)
for s in subs[:22]:
    d = load_sim(s); tr = d["truth"]; n = d["n"]
    pp1.append(np.mean((tr["tnp"] == np.arange(n) + 1) | (tr["tnp"] < 0)))
    for L in range(4):
        m = d["lim"] == L; anr[L] += d["anch"][m].sum(); anc[L] += m.sum()
        su, cf = d["succ"][L], d["conf"][L]; tn = tr["tnode"][L]
        ok = su >= 0
        corr = np.zeros(n, bool); corr[ok] = tr["tnp"][tn[ok]] == tn[su[ok]]
        b = np.digitize(cf, bins) - 1
        for k in range(len(bins) - 1):
            mm = ok & (b == k); acc[L, k] += corr[mm].sum(); cnt[L, k] += mm.sum()
    for t in fr:
        frag, off, flim, flen, fn = fragments(d["succ"], d["conf"], t)
        # fragment correct: all internal links correct
        good = []
        for f, path in enumerate(fn):
            L = flim[f]; tp = tr["tnode"][L][path]
            good.append(np.all(tr["tnp"][tp[:-1]] == tp[1:]) if len(path) > 1 else True)
        fr[t].append((flen, np.array(good), flim))
print("tnp is p+1 or -1:", np.round(pp1, 4).min())
print("anchored share per limb", np.round(anr / anc, 4))
print("chain link precision by conf bin (rows limb) bins", bins)
for L in range(4):
    print(L, " ".join(f"{acc[L,k]/max(cnt[L,k],1):.3f}({cnt[L,k]/cnt[L].sum():.2f})" for k in range(len(bins) - 1)))
for t in fr:
    flen = np.concatenate([x[0] for x in fr[t]]); good = np.concatenate([x[1] for x in fr[t]]); flim = np.concatenate([x[2] for x in fr[t]])
    for L in range(4):
        m = flim == L
        w = flen[m]
        print(f"thr {t} limb {L}: frags {m.sum()} mean len {w.mean():.1f} node-weighted mean len {np.sum(w*w)/w.sum():.1f} "
              f"share nodes in len>=10 {w[w>=10].sum()/w.sum():.3f} nodes in internally-correct frags {w[good[m]].sum()/w.sum():.3f}")
