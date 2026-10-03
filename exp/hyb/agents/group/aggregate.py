"""Aggregate eval_<tag>.json into the per-limb-pair simulation table (mean over sessions, sample-weighted)."""
import sys, os, json, numpy as np
HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
tag = sys.argv[1] if len(sys.argv) > 1 else "pair_sec"
R = json.load(open(os.path.join(HERE, f"eval_{tag}.json")))
sessions = [k for k in R if k.startswith("sbj_")]
pairs = list(R[sessions[0]]["pairs"].keys())
print(f"sessions: {sessions}  N: {[R[s]['N'] for s in sessions]}  chance same-label: {[round(R[s]['chance_same'],3) for s in sessions]}")
print(f"{'limb pair':22s} | {'exact all':>9s} {'exact dyn':>9s} {'exact sta':>9s} | {'same all':>8s} {'same dyn':>8s} {'same sta':>8s} | n_dyn n_sta")
for p in pairs:
    acc = {}
    for k in ("all", "dyn", "sta"):
        n = np.array([R[s]["pairs"][p][k]["n"] for s in sessions], float)
        ex = np.array([R[s]["pairs"][p][k]["exact"] for s in sessions]); sm = np.array([R[s]["pairs"][p][k]["same"] for s in sessions])
        acc[k] = ((ex * n).sum() / n.sum(), (sm * n).sum() / n.sum(), int(n.sum()))
    print(f"{p:22s} | {acc['all'][0]:9.3f} {acc['dyn'][0]:9.3f} {acc['sta'][0]:9.3f} | {acc['all'][1]:8.3f} {acc['dyn'][1]:8.3f} {acc['sta'][1]:8.3f} | {acc['dyn'][2]} {acc['sta'][2]}")
n = np.array([R[s]["N"] for s in sessions], float)
ch = np.array([R[s]["chance_same"] for s in sessions])
print(f"chance same-label (random pairing), weighted: {(ch*n).sum()/n.sum():.3f}")
g4 = {k: (np.array([R[s]["group4"][k] for s in sessions]) * n).sum() / n.sum() for k in ("exact4", "same4", "pairwise_same")}
print("4-way consistent grouping (legs L-R, arms L-R, legpair-armpair):", {k: round(v, 3) for k, v in g4.items()})
for k in ("all", "dyn", "sta"):
    nn = np.array([R[s]["anchor"][k]["n"] for s in sessions], float)
    ex = (np.array([R[s]["anchor"][k]["exact"] for s in sessions]) * nn).sum() / nn.sum()
    sm = (np.array([R[s]["anchor"][k]["same"] for s in sessions]) * nn).sum() / nn.sum()
    print(f"anchor-centric {k}: exact {ex:.3f} same-label {sm:.3f} (n={int(nn.sum())})")
print("per-session anchor-centric same-label:", {s: round(R[s]["anchor"]["all"]["same"], 3) for s in sessions})
