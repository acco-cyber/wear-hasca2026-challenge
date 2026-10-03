"""Parse the 2nd WEAR challenge test.csv (id, sbj_id, sensor_location, x_axis, y_axis, z_axis; each axis cell a list of
50 numbers) into work/w25/w25.npz: acc (N,50,3) float32, id, sbj, limb (our order la,ll,ra,rl = 0..3), and print a
format report (rows per subject/limb, id ordering hints, masked-axis share, duplicate/overlap structure)."""
import os, sys, re, io, zipfile, time
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"; SRC = os.path.join(W, "data", "wear2025"); OUT = os.path.join(W, "work", "w25")
LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
os.makedirs(OUT, exist_ok=True)

def find():
    for f in os.listdir(SRC):
        p = os.path.join(SRC, f)
        if f.lower().endswith(".zip"):
            z = zipfile.ZipFile(p); name = [n for n in z.namelist() if n.endswith("test.csv")][0]
            return io.TextIOWrapper(z.open(name), encoding="utf-8")
        if f.lower() == "test.csv":
            return open(p, encoding="utf-8")
    raise SystemExit("test.csv not found in " + SRC)

def parse_axis(col):
    s = col.str.replace(r"[\[\]\n]", " ", regex=True).str.replace(",", " ")
    return np.array([np.array(v.split(), dtype=np.float32) for v in s.to_numpy()])

t0 = time.time(); df = pd.read_csv(find()); print("columns", df.columns.tolist(), "rows", len(df), f"[{time.time() - t0:.0f}s]")
print(df.head(3).to_string()[:1200])
ax = [parse_axis(df[c].astype(str)) for c in ("x_axis", "y_axis", "z_axis")]
print("axis shapes", [a.shape for a in ax])
acc = np.stack(ax, 2).astype(np.float32)
sbj = df["sbj_id"].to_numpy(); sbj = np.array([int(str(s).replace("sbj_", "")) for s in sbj])
limb = np.array([LIMBS.index(str(l).strip().lower().replace(" ", "_")) for l in df["sensor_location"]])
ids = df["id"].to_numpy()
np.savez(os.path.join(OUT, "w25.npz"), acc=acc, id=ids, sbj=sbj, limb=limb)
print("saved", acc.shape, f"[{time.time() - t0:.0f}s]")
print("rows per subject:", dict(zip(*np.unique(sbj, return_counts=True))))
print("rows per (subject, limb):"); print(pd.crosstab(sbj, limb))
print("ids: min", ids.min(), "max", ids.max(), "sorted?", bool((np.diff(ids) > 0).all()))
zero_axis = (np.abs(acc).max(1) == 0); print("share of windows with an all-zero axis:", zero_axis.any(1).mean().round(3), "per axis", zero_axis.mean(0).round(3))
nan = ~np.isfinite(acc).all((1, 2)); print("windows with NaN:", int(nan.sum()))
mag = np.linalg.norm(acc, axis=2); print("magnitude mean per window quantiles (g):", np.quantile(mag.mean(1), [0.05, 0.5, 0.95]).round(3))
# consecutive-row overlap hint: does row i+1 look like row i shifted by k samples (unshuffled sliding windows)?
m = mag[~zero_axis.any(1)]
for k in (1, 5, 10, 25, 50):
    a, b = m[:-1, k:], m[1:, :50 - k]
    if a.shape[1] >= 5:
        c = [np.corrcoef(a[i], b[i])[0, 1] for i in range(0, min(len(a), 3000), 7)]
        print(f"adjacent-row shift {k}: median corr {np.nanmedian(c):.3f}")
