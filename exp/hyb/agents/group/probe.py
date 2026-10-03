import sys, time, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import *

name = sys.argv[1] if len(sys.argv) > 1 else "sbj_5"
t0 = time.time()
acc, lab = load_session(name)
nsec = acc.shape[1]
print(name, "seconds", nsec, "nan frac", np.isnan(acc).mean(), "load", round(time.time() - t0, 1))
# inter-limb lag check on dynamic seconds: full-session hp magnitude cross-correlation
mag = np.sqrt((np.nan_to_num(acc) ** 2).sum(-1)).reshape(4, -1)
hp = mag - np.convolve(np.ones(25) / 25, np.ones(1))[0] * 0  # placeholder
hp = mag - mag.mean(1, keepdims=True)
for (a, b) in PAIRS:
    cc = [np.corrcoef(hp[a, 10:-10], np.roll(hp[b], l)[10:-10])[0, 1] for l in range(-5, 6)]
    print(LIMBS[a], LIMBS[b], "lag argmax", np.argmax(cc) - 5, "corr", np.round(cc, 3))
D = [window_desc(acc[l]) for l in range(4)]
dyn = D[0]["dyn"] | D[1]["dyn"] | D[2]["dyn"] | D[3]["dyn"]
print("dyn frac per limb", [round(float(d["dyn"].mean()), 3) for d in D], "any", round(float(dyn.mean()), 3))
truth = np.arange(nsec)
for (a, b) in PAIRS:
    t1 = time.time()
    M = pair_matrices(D[a], D[b])
    S = simple_score(M)
    # top-1 accuracy (argmax) and hungarian
    top1 = (S.argmax(1) == truth).mean()
    asg = hungarian(S)
    r = eval_assignment(asg, truth, lab, lab, D[a]["dyn"])
    print(f"{LIMBS[a]:9s}-{LIMBS[b]:9s} top1 {top1:.3f} | exact all {r['all'][0]:.3f} dyn {r['dyn'][0]:.3f} sta {r['sta'][0]:.3f} | same-label all {r['all'][1]:.3f} dyn {r['dyn'][1]:.3f} sta {r['sta'][1]:.3f} | n_dyn {r['dyn'][2]} n_sta {r['sta'][2]} | {time.time()-t1:.1f}s")
    # per-feature top1 for insight
    if (a, b) == (1, 3):
        for k, v in M.items():
            print("   feat", k, "top1", round(float((v.argmax(1) == truth).mean()), 3), "dyn", round(float((v.argmax(1) == truth)[D[a]["dyn"]].mean()), 3))
