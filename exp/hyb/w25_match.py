"""Match our 2026 test tiles to their augmented twins in the 2025 set (same subject, same limb, same second) and probe
the 2025 id structure. Writes work/w25/match.npz: twin (2026 row -> 2025 row), corr, margin."""
import os, sys, time
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"; OUT = os.path.join(W, "work", "w25")
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
z = np.load(os.path.join(OUT, "w25.npz")); A25 = z["acc"].astype(np.float32); id25 = z["id"]; s25 = z["sbj"]; l25 = z["limb"]
A26 = np.load(os.path.join(W, "data", "test", "test_inertial_data.npy")).astype(np.float32)
tm = pd.read_csv(os.path.join(W, "data", "test", "test_meta_data.csv")); s26 = tm.sbj_id.to_numpy(); l26 = np.array([LIMBS.index(x) for x in tm.sensor_location])
print("2026", A26.shape, "2025", A25.shape)
# 1) id structure of the 2025 file
runs = np.flatnonzero(np.diff(s25 * 4 + l25) != 0); print("2025 rows: (subject,limb) changes along id:", len(runs), "-> blocks of", np.round(np.diff(np.r_[0, runs + 1, len(s25)]).mean(), 1), "rows")
def zs(m):
    m = m - m.mean(1, keepdims=True); return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-6)
m25 = np.linalg.norm(A25, axis=2); m26 = np.linalg.norm(A26, axis=2)
# continuity between consecutive 2025 rows inside a block (time-ordered blocks would make this small)
same = (np.diff(s25 * 4 + l25) == 0); jump = np.abs(m25[1:, 0] - m25[:-1, -1]); rnd = np.abs(m25[1:, 0] - np.roll(m25[:-1, -1], 7))
print(f"2025 consecutive-row magnitude jump median {np.median(jump[same]):.3f} vs random pairs {np.median(rnd[same]):.3f}")
# 2) twin matching per (subject, limb) via magnitude correlation (rotation/sign invariant, scale invariant)
twin = np.full(len(A26), -1, np.int64); corr = np.zeros(len(A26), np.float32); margin = np.zeros(len(A26), np.float32)
t0 = time.time()
for s in np.unique(s26):
    for l in range(4):
        i26 = np.flatnonzero((s26 == s) & (l26 == l)); i25 = np.flatnonzero((s25 == s) & (l25 == l))
        C = zs(m26[i26]) @ zs(m25[i25]).T                         # (n26, n25)
        best = C.argmax(1); top = C[np.arange(len(i26)), best]
        C[np.arange(len(i26)), best] = -2; second = C.max(1)
        twin[i26] = i25[best]; corr[i26] = top; margin[i26] = top - second
        print(f"sbj {s} limb {LIMBS[l]}: n26 {len(i26)} n25 {len(i25)} corr median {np.median(top):.4f} | >0.99 {np.mean(top > 0.99):.3f} | margin>0.1 {np.mean(top - second > 0.1):.3f} | unique twins {len(np.unique(best))}/{len(i26)} [{time.time() - t0:.0f}s]")
np.savez(os.path.join(OUT, "match.npz"), twin=twin, corr=corr, margin=margin)
# 3) id relations
ok = corr > 0.99
print(f"\nconfident twins: {ok.mean():.3f}")
d = id25[twin[ok]]
print("2025 id of twin vs 2026 id: corr", np.corrcoef(d, np.arange(len(A26))[ok])[0, 1].round(4), "| twin_id mod 4 == limb?", np.mean((d % 4) == l26[ok]).round(3), "| twin_id//4 == 2026 id?", np.mean((d // 4) == np.arange(len(A26))[ok]).round(3))
# per subject: are 2025 ids of our tiles' twins monotone in anything we know? (we don't know time; report spread)
for s in np.unique(s26):
    m = ok & (s26 == s); print(f"sbj {s}: twin ids min {id25[twin[m]].min()} max {id25[twin[m]].max()}")
# augmentation estimate from confident twins: scale factor and whether axes are permuted/flipped
sc = (m25[twin[ok]].mean(1) / m26[ok].mean(1)); print("scale factor quantiles", np.quantile(sc, [0.01, 0.1, 0.5, 0.9, 0.99]).round(3))
res = []
for i in np.flatnonzero(ok)[:2000]:
    a, b = A26[i], A25[twin[i]]; R = np.linalg.lstsq(a, b, rcond=None)[0]; res.append(np.abs(b - a @ R).std())
print("rotation-fit residual std (g) median", np.median(res).round(4), "90%", np.quantile(res, 0.9).round(4))
