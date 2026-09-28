"""Align the Hanbat notebook's OOF tile rows (keep/sim_meta.npz: rec, start, sensor, y, sbj, fold) with our
data/prep/train_meta.csv rows (session, t), and export our OOF base probabilities (limb-selected to their drawn sensor)
in THEIR row order, plus the matching full-data test probabilities.
python align_rows.py <keep_dir>  -> exp/hyb/rows.npz (their_to_ours, ours_to_theirs, their sensor as our limb index)
                                    exp/hyb/base_<name>_oof.npy (69326,19), base_<name>_test.npy (12234,19)"""
import os, sys, json
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb")
KEEP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(W, "work", "hanbat", "keep")
THEIR_SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]      # their sensor index
OUR_LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]           # our OOF axis-1 order
S2L = np.array([OUR_LIMBS.index(s) for s in THEIR_SENSORS])              # their sensor -> our limb index
BASES = {  # name: (oof (69326,4,19) in our rows, test (12234,19))
    "v3b": (os.path.join(W, "exp", "base", "v3b", "oof.npy"), os.path.join(W, "exp", "full", "v3b_full", "test.npy")),
    "v1": (os.path.join(W, "work", "lgbm_v1", "oof.npy"), os.path.join(W, "exp", "full", "v1_full", "test.npy")),
    "fusion": (os.path.join(W, "work", "fusion_v1", "oof.npy"), os.path.join(W, "work", "fusion_v1", "test.npy")),
    "lgbm_v2": (os.path.join(W, "work", "lgbm_v2", "oof.npy"), None),
    # 5-fold UEC members from kernel wear-uec-k3 (OOF already in our train_meta layout; see kaggle/uec_k3)
    "k3cnn8": (os.path.join(W, "work", "uec_k3", "oof_cnn8.npy"), os.path.join(W, "work", "uec_k3", "test_cnn8.npy")),
    "k3xcep": (os.path.join(W, "work", "uec_k3", "oof_xcep.npy"), os.path.join(W, "work", "uec_k3", "test_xcepvid.npy")),
    # 5-fold UEC inertial LightGBM (uec/gbdt/train_uec.py std --tile_only --no_es), OOF assembled by uec/gbdt/collect_std.py
    "ueclgbm": (os.path.join(W, "uec", "gbdt", "std_cv", "oof.npy"), os.path.join(W, "uec", "gbdt", "std_cv", "test.npy")),
}

def main():
    sm = {k: v for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    meta = pd.read_csv(os.path.join(W, "data", "prep", "train_meta.csv"))
    stems = sorted(meta.session.unique())          # same order as sorted(Path.glob('*.csv')) stems
    assert len(stems) == 24
    key_ours = {(s, int(t)): i for i, (s, t) in enumerate(zip(meta.session, meta.t))}
    their_to_ours = np.array([key_ours[(stems[r], int(st) // 50)] for r, st in zip(sm["rec"], sm["start"])])
    assert len(np.unique(their_to_ours)) == len(their_to_ours) == len(meta), "row sets differ"
    assert (sm["start"] % 50 == 0).all()
    y_ours = meta.y.to_numpy()[their_to_ours]
    print("rows", len(their_to_ours), "label agreement with our y:", np.mean(y_ours == sm["y"]))
    print("their folds:", {f: sorted(set(sm["sbj"][sm["fold"] == f].tolist())) for f in range(5)})
    ours_to_theirs = np.empty_like(their_to_ours); ours_to_theirs[their_to_ours] = np.arange(len(their_to_ours))
    limb = S2L[sm["sensor"].astype(int)]
    os.makedirs(HYB, exist_ok=True)
    np.savez(os.path.join(HYB, "rows.npz"), their_to_ours=their_to_ours, ours_to_theirs=ours_to_theirs, limb=limb)
    for name, (po, pt) in BASES.items():
        if not os.path.exists(po):
            print(name, "missing"); continue
        o = np.load(po).astype(np.float32)[their_to_ours]                 # (N,4,19) in their order
        P = o[np.arange(len(o)), limb]
        bad = ~np.isfinite(P).all(1) | (np.nansum(P, 1) <= 0)
        P = np.where(bad[:, None], 1.0 / 19, P); P = P / P.sum(1, keepdims=True)
        acc = np.mean(P.argmax(1) == sm["y"])
        cm = np.bincount(sm["y"] * 19 + P.argmax(1), minlength=361).reshape(19, 19); tp = np.diag(cm)
        f1 = np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1))
        print(f"{name}: OOF rows with NaN limb {bad.sum()}, window acc {acc:.4f} macroF1 {f1:.4f}")
        np.save(os.path.join(HYB, f"base_{name}_oof.npy"), P.astype(np.float32))
        if pt and os.path.exists(pt):
            T = np.load(pt).astype(np.float32); T = np.clip(T, 0, None); T /= T.sum(1, keepdims=True)
            np.save(os.path.join(HYB, f"base_{name}_test.npy"), T)
    print("done")

if __name__ == "__main__":
    main()
