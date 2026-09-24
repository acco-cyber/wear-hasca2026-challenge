"""TEST link structure with the cross-limb successor scorer (same format as work/test_structure.pkl).
python build_test.py <model_tag> [out.pkl]    (model = models_<tag>.pkl from train2.py; xl ridge = xl_models.pkl)
Graph log-odds = successor log-odds (variant "S"); chains = linear assignment on the same log-odds."""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xl_common import *
from train2 import predict_lo, KNN_NAMES

if __name__ == "__main__":
    tags = sys.argv[1].split("+"); out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(XD, f"test_structure_xl_{sys.argv[1]}.pkl")
    Ms = [pickle.load(open(os.path.join(XD, f"models_{t}.pkl"), "rb")) for t in tags]; xl = pickle.load(open(os.path.join(XD, "xl_models.pkl"), "rb"))
    tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv")); assert (tm.id.to_numpy() == np.arange(len(tm))).all()
    xi = np.load(os.path.join(DATA, "test", "test_inertial_data.npy")).astype(np.float16).astype(np.float32)   # match train quantisation
    vid = np.load(os.path.join(PREP, "test_vid_pca.npy"), mmap_mode="r"); limb_all = np.array([LIMBS.index(l) for l in tm.sensor_location])
    need_knn = any(M.get("knn") for M in Ms)
    if need_knn:
        from knnx import Bank, knn_feats
        meta, imu, _, sl = load_prep(); bank = Bank(imu, sl)
    S = {}
    for s in sorted(tm.sbj_id.unique()):
        t0 = time.time(); idx = np.where(tm.sbj_id.to_numpy() == s)[0]; n = len(idx); limb = limb_all[idx]
        cand, F = all_pair_features(np.asarray(vid[idx], np.float32), xi[idx], limb, xl)
        F = np.concatenate([F, knn_feats(xi[idx], limb, cand, bank) if need_knn else np.full(F.shape[:2] + (len(KNN_NAMES),), np.nan, np.float32)], 2)
        lo = np.mean([predict_lo(M["ms"], F[..., M["cols"]]) for M in Ms], 0).astype(np.float32)
        succ0, sc, Lm = assignment(cand, lo, n)
        S[s] = dict(idx=idx, cand=cand, lo=lo, succ0=succ0, sc=sc, Lm=Lm.astype(np.float16))
        print(f"sbj {s}: n={n} ({time.time()-t0:.0f}s) median top logodds {np.median(lo.max(1)):.2f}", flush=True)
    pickle.dump(S, open(out, "wb")); print("wrote", out)
