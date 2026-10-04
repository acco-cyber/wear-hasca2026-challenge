"""Heavier variants of the v4 fork (patch_v4.py) that spend more GPU inside one coherent run:
  * window models: 8 fusion + 4 IMU-only seeds ("pool"), or 6 pool + 2 transformer-over-frames fusion seeds + 3 IMU ("tf");
  * two-tower boundary matcher: four seeds (two at a time, one per GPU), each kept as its own link-scorer features;
  * link scorer: five LightGBM seeds instead of three;
  * the raw outputs of every window run are kept (keep4/runs/) for offline blend studies.
  python patch_big.py pool 7    -> kaggle/v4_fork/big-pool-s7/
  python patch_big.py tf 9      -> kaggle/v4_fork/big-tf-s9/"""
import sys
from patch_v4 import patch

IMU1 = '    ("imu_s1", "--seed 1 --epochs 30 --eval_every 5 --only imu --knn 1,3,5"),\n]'
ADD = {
    "pool": ['    ("final_s4", "--seed 4 --epochs 12 --eval_every 4 --knn 1,3,5"),',
             '    ("final_s5", "--seed 5 --epochs 12 --eval_every 4 --knn 1,3,5"),',
             '    ("imu_s2", "--seed 2 --epochs 30 --eval_every 5 --only imu --knn 1,3,5"),',
             '    ("final_s6", "--seed 6 --epochs 12 --eval_every 4 --knn 1,3,5"),',
             '    ("final_s7", "--seed 7 --epochs 12 --eval_every 4 --knn 1,3,5"),',
             '    ("imu_s3", "--seed 3 --epochs 30 --eval_every 5 --only imu --knn 1,3,5"),'],
    "tf": ['    ("final_s4", "--seed 4 --epochs 12 --eval_every 4 --knn 1,3,5"),',
           '    ("tfv_s0", "--seed 0 --epochs 12 --eval_every 4 --knn 1,3,5 --vid_arch tf"),',
           '    ("imu_s2", "--seed 2 --epochs 30 --eval_every 5 --only imu --knn 1,3,5"),',
           '    ("final_s5", "--seed 5 --epochs 12 --eval_every 4 --knn 1,3,5"),',
           '    ("tfv_s1", "--seed 1 --epochs 12 --eval_every 4 --knn 1,3,5 --vid_arch tf"),'],
}
FUS_OLD = '    FUSION = [RUNS_DIR / n for n in ("final_s0", "final_s1", "final_s2", "final_s3") if run_complete(n)]'
FUS_NEW = ('    FUSION = [RUNS_DIR / n for n in ("final_s0", "final_s1", "final_s2", "final_s3", "final_s4", "final_s5", "final_s6", "final_s7",\n'
           '                                     "tfv_s0", "tfv_s1") if run_complete(n)]   # FORK: more fusion seeds')
IMU_OLD = '    IMU = [RUNS_DIR / n for n in ("imu_s0", "imu_s1") if run_complete(n)]'
IMU_NEW = '    IMU = [RUNS_DIR / n for n in ("imu_s0", "imu_s1", "imu_s2", "imu_s3") if run_complete(n)]   # FORK'
BUDGET = ('WINDOW_BUDGET_MIN = float(os.environ.get("WEAR_WINDOW_BUDGET_MIN", "150"))',
          'WINDOW_BUDGET_MIN = float(os.environ.get("WEAR_WINDOW_BUDGET_MIN", "330"))   # FORK: more window runs')
LINKS = ('    LINK_DIRS = [ROOT / "link", ROOT / "link_s1"]',
         '    LINK_DIRS = [ROOT / "link"] + [ROOT / f"link_s{_k}" for _k in range(1, 4)]   # FORK: four seeds')
SCHED = ('        jobs = [(out, run_link(out, _seed0 + k, "0,1,2,3,4", k % 2)) for k, out in todo]',
         '        jobs = []                                   # FORK: two seeds at a time, one per GPU\n'
         '        for _i in range(0, len(todo), 2):\n'
         '            _pair = [(out, run_link(out, _seed0 + k, "0,1,2,3,4", _j)) for _j, (k, out) in enumerate(todo[_i:_i + 2])]\n'
         '            for _o, _p in _pair:\n'
         '                _p.wait()\n'
         '            jobs += _pair')
LKS = ("LK_SEEDS = 1 if SMOKE else 3", "LK_SEEDS = 1 if SMOKE else 5   # FORK")
RUNS_KEEP = ('print("FORK kept:"',
             '_RK = KEEP / "runs"; _RK.mkdir(exist_ok=True)       # FORK: raw outputs of every window run\n'
             'for _r in sorted(RUNS_DIR.iterdir()):\n'
             '    if _r.is_dir() and run_complete(_r.name):\n'
             '        _z = np.load(_r / "oof_raw.npz")\n'
             '        np.savez(_RK / f"{_r.name}.npz", oof_logp=_z["logp"].astype(np.float16), oof_emb=_z["emb"].astype(np.float16),\n'
             '                 test_logp=np.load(_r / "test_logp_raw.npy").astype(np.float16), test_emb=np.load(_r / "test_emb.npy").astype(np.float16))\n'
             'print("FORK kept:"')


# "opt": the decode itself uses a per-subject whitened kNN embedding (each subject's own covariance, shrunk 0.1, power
# 0.5, kNN temperature 0.3) and the exercise identity in the count regressor, in stage A (-> better pseudo-labels for the
# adapted expert T) and in stage B. Local OOF on a v2 fit: 0.9255 -> 0.9296 (per-subject whitening), public LB 0.92758.
OPT = [
    ('KNN_WHITEN = float(os.environ.get("WEAR_WHITEN", "0"))', 'KNN_WHITEN = float(os.environ.get("WEAR_WHITEN", "0.5"))   # FORK: on', 1),
    ('KNN_EMB_OOF, KNN_EMB_TEST = (whitened(d["emb"], d["sbj"], t["emb"], t["sbj"]) if KNN_WHITEN > 0 else (d["emb"], t["emb"]))',
     '''def whitened_subject(E, sb, power=KNN_WHITEN, shrink=0.1):
    """FORK: every subject (training or test) centred and whitened with its OWN covariance, shrunk to a scaled identity"""
    out = np.empty(E.shape, np.float32)
    for s in np.unique(sb):
        ii = np.flatnonzero(sb == s)
        X = E[ii].astype(np.float64)
        X -= X.mean(0)
        C = np.cov(X.T)
        C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        ev, U = np.linalg.eigh(C)
        out[ii] = (X @ (U / np.maximum(ev, 1e-8) ** power)).astype(np.float32)
    return out


KNN_EMB_OOF, KNN_EMB_TEST = ((whitened_subject(d["emb"], d["sbj"]), whitened_subject(t["emb"], t["sbj"])) if KNN_WHITEN > 0
                             else (d["emb"], t["emb"]))''', 1),
    ('        X, key = profile_features([P, Bp], sbj, TRAIN_SETS)\n',
     '        X, key = profile_features([P, Bp], sbj, TRAIN_SETS)\n'
     '        X = np.concatenate([X, np.eye(N_CLS - 1)[key[:, 1] - 1]], 1)      # FORK: exercise identity\n', 1),
    ('        Xt, kt = profile_features([Pt, Btp], t["sbj"], {})\n',
     '        Xt, kt = profile_features([Pt, Btp], t["sbj"], {})\n'
     '        Xt = np.concatenate([Xt, np.eye(N_CLS - 1)[kt[:, 1] - 1]], 1)     # FORK: exercise identity\n', 1),
]


def build(kind, seed, opt=False):
    extra = [(IMU1, IMU1[:-1] + "\n".join(ADD[kind]) + "\n]", 1), (FUS_OLD, FUS_NEW, 1), (IMU_OLD, IMU_NEW, 1), (*BUDGET, 1),
             (*LINKS, 1), (*SCHED, 1), (*LKS, 1), (*RUNS_KEEP, 1)] + (OPT if opt else [])
    return patch(seed, extra, name=f"v4-big-{kind}{'-opt' if opt else ''}-s{seed}")


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), opt=len(sys.argv) > 3 and sys.argv[3] == "opt")
