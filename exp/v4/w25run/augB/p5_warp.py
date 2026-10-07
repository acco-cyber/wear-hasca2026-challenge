"""probe 5: endpoint-hit pairs -> signed permutation P and the time-warp tau(p) per pair; is the warp shared?"""
import os, csv, itertools, numpy as np
np.set_printoptions(linewidth=220, suppress=True)
W = r"E:\Claude code\wear"; D = r"E:\Claude code\wear\exp\v4\w25run\augB"
z = np.load(os.path.join(W, "work", "w25", "w25.npz")); A25 = z["acc"].astype(np.float64); s25 = z["sbj"]; l25 = z["limb"]
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
m = np.load(os.path.join(W, "work", "w25", "match.npz")); twin = m["twin"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float64)
rows = list(csv.DictReader(open(os.path.join(W, "data", "test", "test_meta_data.csv"))))
s26 = np.array([int(r["sbj_id"]) for r in rows]); l26 = np.array([LIMBS.index(r["sensor_location"]) for r in rows])
ex = np.abs(A26 - A25[twin]).max((1, 2)) < 1e-6
bad = np.flatnonzero(~ex)
# signed permutations
SP = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        P = np.zeros((3, 3)); P[list(perm), range(3)] = sg; SP.append(P)
SP = np.array(SP)  # (48,3,3): b = a @ P
# endpoint hash over 2025 rows: key of sample 0 and sample 49 (perm/sign invariant)
def key(v):
    return tuple(np.round(np.sort(np.abs(v)), 4))
H = {}
for r in range(len(A25)):
    H.setdefault((key(A25[r, 0]), key(A25[r, 49])), []).append(r)
pairs = []
for i in bad:
    c = [r for r in H.get((key(A26[i, 0]), key(A26[i, 49])), []) if s25[r] == s26[i]]
    if c:
        pairs.append((i, c[0], len(c)))
print("non-exact", len(bad), "endpoint-hit pairs", len(pairs), "multi", sum(p[2] > 1 for p in pairs))
print("  same limb", np.mean([l25[r] == l26[i] for i, r, _ in pairs]).round(3))


def solve_tau(a, b):
    """b[p] ~ a(tau_p) with piecewise-linear a; returns tau and the residual"""
    tau = np.zeros(50); err = np.zeros(50)
    for p in range(50):
        best = (1e9, 0.0)
        for k in range(49):
            d = a[k + 1] - a[k]; dd = d @ d
            f = 0.0 if dd < 1e-12 else np.clip((b[p] - a[k]) @ d / dd, 0, 1)
            e = np.linalg.norm(a[k] + f * d - b[p])
            if e < best[0] - 1e-12:
                best = (e, k + f)
        err[p], tau[p] = best
    return tau, err


out = []
pcode = []
for i, r, _ in pairs:
    a, b = A26[i], A25[r]
    e = [np.abs(a[[0, 49]] @ P - b[[0, 49]]).max() for P in SP]; j = int(np.argmin(e)); P = SP[j]
    tau, err = solve_tau(a @ P, b)
    out.append(tau); pcode.append(j)
    if len(out) <= 3 or err.max() > 0.01 and len(out) < 40:
        pass
out = np.array(out); pcode = np.array(pcode)
print("signed-perm usage:")
for j, c in zip(*np.unique(pcode, return_counts=True)):
    print("  P=", SP[j].astype(int).tolist(), "count", c, "| limbs", {LIMBS[x]: int(np.sum((pcode == j) & (np.array([l26[i] for i, _, _ in pairs]) == x))) for x in range(4)})
med = np.median(out, 0); dev = np.abs(out - med).max(1)
print("median tau\n", med.round(3))
print("max |tau - median| per pair quantiles", np.quantile(dev, [0.5, 0.9, 0.99, 1]).round(4))
np.savez(os.path.join(D, "p5_pairs.npz"), pairs=np.array([(i, r) for i, r, _ in pairs]), tau=out, pcode=pcode, SP=SP)
