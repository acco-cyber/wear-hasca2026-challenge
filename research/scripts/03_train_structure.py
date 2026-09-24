"""Task (5): train label structure (bouts, durations, null fraction, order), video frame count vs inertial length."""
import numpy as np, pandas as pd, os, sys, glob
D = r"E:\Claude code\wear\data\train"
A = r"E:\Claude code\wear\research\artifacts"
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)", "jogging (butt-kicks)",
           "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)", "stretching (lumbar rotation)",
           "push-ups", "push-ups (complex)", "sit-ups", "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
want = sys.argv[1:] if len(sys.argv) > 1 else ["0", "5", "20"]
avail = sorted(glob.glob(os.path.join(D, "inertial_feat", "sbj_*.csv")))
print("available train CSVs:", [os.path.basename(p) for p in avail])
rows = []; orders = {}
for s in want:
    p = os.path.join(D, "inertial_feat", f"sbj_{s}.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print("missing", p); continue
    df = pd.read_csv(p)
    print(f"\n=== sbj_{s}: shape {df.shape}, cols {list(df.columns)[:3]}..{list(df.columns)[-2:]}")
    print("  sbj_id values:", df["sbj_id"].unique()[:5])
    lab = df["label"].fillna("null").astype(str).values
    print("  unique labels:", sorted(set(lab)))
    print("  NaN in label col:", df["label"].isna().sum(), "| NaN in acc:", df.iloc[:, 1:13].isna().sum().sum())
    n = len(lab); print(f"  samples {n} = {n/50:.1f}s = {n/50/60:.2f} min @50Hz")
    # run-length encoding
    ch = np.where(lab[1:] != lab[:-1])[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [n]])
    runs = pd.DataFrame(dict(label=lab[st], start=st, end=en)); runs["dur_s"] = (runs.end - runs.start) / 50
    runs["sbj"] = s
    rows.append(runs)
    nonnull = runs[runs.label != "null"]
    print("  null fraction: %.3f ; #runs total %d ; #non-null bouts %d ; classes present %d" % ((lab == "null").mean(), len(runs), len(nonnull), nonnull.label.nunique()))
    bc = nonnull.groupby("label").agg(n_bouts=("dur_s", "size"), total_s=("dur_s", "sum"), min_s=("dur_s", "min"), max_s=("dur_s", "max"))
    print(bc.to_string())
    print("  activity order:", list(nonnull.label.values))
    orders[s] = list(nonnull.label.values)
    # null gaps between activities
    nulls = runs[runs.label == "null"]
    print("  null runs: n=%d, dur median %.1fs, min %.1f, max %.1f; first %.1f last %.1f" % (len(nulls), nulls.dur_s.median(), nulls.dur_s.min(), nulls.dur_s.max(), nulls.dur_s.iloc[0], nulls.dur_s.iloc[-1]))
    # short bouts (<3s) = label glitches?
    print("  bouts < 3s:", (runs.dur_s < 3).sum(), "| non-null bouts <3s:", (nonnull.dur_s < 3).sum())
    # video npy check
    vp = os.path.join(D, "videomae_feat", f"sbj_{s}.npy")
    if os.path.exists(vp) and os.path.getsize(vp) > 0:
        try:
            V = np.load(vp, mmap_mode="r")
            print(f"  video npy shape {V.shape} {V.dtype}: frames/30 = {V.shape[0]/30:.2f}s vs inertial {n/50:.2f}s ; ratio frames/samples = {V.shape[0]/n:.4f} (expect 0.6)")
        except Exception as e:
            print("  video npy not readable yet:", e)
    else:
        print("  video npy not available")
if rows:
    allr = pd.concat(rows); allr.to_csv(os.path.join(A, "train_label_runs.csv"), index=False)
    # order comparison
    ks = list(orders)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = orders[ks[i]], orders[ks[j]]
            print(f"order sbj_{ks[i]} == sbj_{ks[j]} ? {a == b}; same set? {set(a)==set(b)}")
    # canonical order (class idx)
    for k in ks:
        print(f"sbj_{k} order idx:", [CLASSES.index(x) if x in CLASSES else -1 for x in orders[k]])
    # summary of bout durations per class over all subjects loaded
    nn = allr[allr.label != "null"]
    print("\nper-class bout duration (s) across loaded subjects:\n", nn.groupby("label").dur_s.describe()[["count", "mean", "min", "50%", "max"]].to_string())
    print("\nnull fraction per subject:\n", allr.groupby("sbj").apply(lambda g: g[g.label == "null"].dur_s.sum() / g.dur_s.sum()).to_string())
