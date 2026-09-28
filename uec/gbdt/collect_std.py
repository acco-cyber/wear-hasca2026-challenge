"""Assemble the std-fold outputs of train_uec.py into our OOF layout: uec/gbdt/std_cv/oof.npy (69326,4,19) over
data/prep/train_meta.csv rows x limbs [left_arm, left_leg, right_arm, right_leg] (NaN where a tile row was not predicted),
and std_cv/test.npy = mean of the fold models' test probabilities."""
import os, glob, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); FEATS = os.path.join(HERE, "feats"); CV = os.path.join(HERE, "std_cv")
OUR_LIMBS = ["left_arm", "left_leg", "right_arm", "right_leg"]
UEC_SENSOR = {"ra": "right_arm", "la": "left_arm", "rl": "right_leg", "ll": "left_leg"}   # feats store the sensor key string
meta = pd.read_csv(r"E:\Claude code\wear\data\prep\train_meta.csv")
key = {(s, int(t)): i for i, (s, t) in enumerate(zip(meta.session, meta.t))}
# row metadata of the feature table, in load_all() order (sorted sbj_*.npz)
sess, start, sensor = [], [], []
for p in sorted(glob.glob(os.path.join(FEATS, "sbj_*.npz"))):
    d = np.load(p); n = len(d["y"]); s = os.path.splitext(os.path.basename(p))[0]
    sess += [s] * n; start.append(d["start"]); sensor.append(d["sensor"])
sess = np.array(sess); start = np.concatenate(start); sensor = np.concatenate(sensor).astype(str)
print("feature rows", len(sess), "sensor values", np.unique(sensor))
oof = np.full((len(meta), 4, 19), np.nan, np.float32); tests = []
for fd in sorted(glob.glob(os.path.join(CV, "fold_*"))):
    if not os.path.exists(os.path.join(fd, "extra_prob.npy")):
        print(fd, "incomplete"); continue
    P = np.load(os.path.join(fd, "extra_prob.npy")); idx = np.load(os.path.join(fd, "extra_idx.npy"))
    rows = np.array([key[(s, int(st) // 50)] for s, st in zip(sess[idx], start[idx])])
    limb = np.array([OUR_LIMBS.index(UEC_SENSOR[str(v)]) for v in sensor[idx]])
    oof[rows, limb] = P; tests.append(np.load(os.path.join(fd, "test_prob.npy")))
    print(fd, "rows", len(idx), json.load(open(os.path.join(fd, "done.json"))).get("val_macro_f1"))
print("OOF filled fraction", np.isfinite(oof[:, :, 0]).mean(), "folds", len(tests))
np.save(os.path.join(CV, "oof.npy"), oof)
if tests:
    np.save(os.path.join(CV, "test.npy"), np.mean(tests, 0).astype(np.float32))
