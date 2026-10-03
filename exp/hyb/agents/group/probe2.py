import sys, time, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import *
from sklearn.metrics import roc_auc_score

name = sys.argv[1] if len(sys.argv) > 1 else "sbj_5"
acc, lab = load_session(name)
nsec = acc.shape[1]
mag = np.sqrt((np.nan_to_num(acc) ** 2).sum(-1)).reshape(4, -1)
# pick a jogging bout
jog = np.where(lab == "jogging")[0]
print("jogging seconds", len(jog), "first run", jog[:5])
# contiguous run
runs = np.split(jog, np.where(np.diff(jog) != 1)[0] + 1)
run = max(runs, key=len)
print("longest jogging run", run[0], run[-1], len(run))
s0, s1 = run[0] * 50, (run[-1] + 1) * 50
seg = mag[:, s0:s1]
seg = seg - seg.mean(1, keepdims=True)
for (a, b) in PAIRS:
    L = 100
    cc = np.array([np.corrcoef(seg[a, L:-L], seg[b, L + l:seg.shape[1] - L + l])[0, 1] for l in range(-L, L + 1)])
    top = np.argsort(-cc)[:5] - L
    print(f"jog {LIMBS[a]:9s}-{LIMBS[b]:9s} lag0 {cc[L]:.3f} best lags {top} vals {np.round(cc[top + L],3)}")
# also for a static-ish bout: stretching (hamstrings)
for act in ["burpees", "push-ups", "stretching (hamstrings)"]:
    idx = np.where(lab == act)[0]
    if len(idx) == 0: continue
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    run = max(runs, key=len)
    s0, s1 = run[0] * 50, (run[-1] + 1) * 50
    seg = mag[:, s0:s1]; seg = seg - seg.mean(1, keepdims=True)
    for (a, b) in [(1, 3), (0, 2), (0, 1)]:
        L = 100
        cc = np.array([np.corrcoef(seg[a, L:-L], seg[b, L + l:seg.shape[1] - L + l])[0, 1] for l in range(-L, L + 1)])
        top = np.argsort(-cc)[:3] - L
        print(f"{act:24s} {LIMBS[a]:9s}-{LIMBS[b]:9s} n={len(run)} lag0 {cc[L]:.3f} best lags {top} vals {np.round(cc[top + L],3)}")

# per-feature AUC: true pairs vs random pairs, dynamic only
D = [window_desc(acc[l]) for l in range(4)]
rng = np.random.default_rng(0)
truth = np.arange(nsec)
for (a, b) in [(1, 3), (0, 2), (0, 1)]:
    M = pair_matrices(D[a], D[b])
    dyn = D[a]["dyn"] & D[b]["dyn"]
    ia = truth[dyn]
    jb = rng.integers(0, nsec, size=(len(ia), 20))
    # same-label negatives
    print(f"--- {LIMBS[a]}-{LIMBS[b]} dyn pairs {len(ia)}")
    for k, v in M.items():
        pos = v[ia, ia]
        neg = v[ia[:, None], jb].ravel()
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        auc = roc_auc_score(y, np.r_[pos, neg])
        # same-label negatives
        print(f"   {k:12s} AUC {auc:.3f}  pos mean {pos.mean():.3f} neg mean {neg.mean():.3f}")
    for k in ["logstd", "fdom"]:
        pos = -np.abs(D[a][k][ia] - D[b][k][ia]); neg = -np.abs(D[a][k][ia][:, None] - D[b][k][jb]).ravel()
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        print(f"   -|d {k}|     AUC {roc_auc_score(y, np.r_[pos, neg]):.3f}")
