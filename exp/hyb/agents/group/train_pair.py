"""Train the LightGBM same-second pair model on the non-eval train sessions.
usage: python train_pair.py <target: sec|lab> <out_name>"""
import sys, os, time, glob, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
import lightgbm as lgb
from feats import pair_rows, PAIR_NAMES, NPF
from common import PAIRS

HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
CACHE = os.path.join(HERE, "cache")
EVAL = {"sbj_5", "sbj_9", "sbj_12", "sbj_20", "sbj_21", "sbj_2"}
VALID = {"sbj_1", "sbj_10", "sbj_17"}
target = sys.argv[1] if len(sys.argv) > 1 else "sec"
out_name = sys.argv[2] if len(sys.argv) > 2 else f"pair_{target}"
N_RAND, N_SAME = 8, 3
rng = np.random.default_rng(1)


def load_cache(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    D = [{k: z[f"{k}{l}"] for k in ("desc", "z_spec", "z_specj", "z_acf", "std", "dyn", "gmean")} for l in range(4)]
    return D, z["lab"]


def build(name):
    D, lab = load_cache(name)
    n = len(lab)
    # same-label pools
    labs, inv = np.unique(lab, return_inverse=True)
    pools = [np.where(inv == k)[0] for k in range(len(labs))]
    Xs, ys, ws = [], [], []
    for pid, (a, b) in enumerate(PAIRS):
        s = np.arange(n)
        ia = [s]; ib = [s]
        jr = rng.integers(0, n, size=(n, N_RAND))
        ia.append(np.repeat(s, N_RAND)); ib.append(jr.ravel())
        js = np.empty((n, N_SAME), dtype=np.int64)
        for i in range(n):
            p = pools[inv[i]]
            js[i] = p[rng.integers(0, len(p), size=N_SAME)]
        ia.append(np.repeat(s, N_SAME)); ib.append(js.ravel())
        ia = np.concatenate(ia); ib = np.concatenate(ib)
        X = pair_rows(D[a], D[b], ia, ib, pid)
        if target == "sec":
            y = (ia == ib).astype(np.float32)
        else:
            y = (lab[ia] == lab[ib]).astype(np.float32)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(CACHE, "sbj_*.npz")))]
tr_names = [n for n in names if n not in EVAL and n not in VALID]
va_names = [n for n in names if n in VALID]
print("train sessions", tr_names); print("valid sessions", va_names, flush=True)
t0 = time.time()
Xtr, ytr = zip(*[build(n) for n in tr_names]); Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
Xva, yva = zip(*[build(n) for n in va_names]); Xva = np.concatenate(Xva); yva = np.concatenate(yva)
print("built", Xtr.shape, ytr.mean(), Xva.shape, yva.mean(), round(time.time() - t0, 1), "s", flush=True)
params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, max_bin=127)
dtr = lgb.Dataset(Xtr, ytr, feature_name=PAIR_NAMES, categorical_feature=["pair"], free_raw_data=True)
dva = lgb.Dataset(Xva, yva, reference=dtr)
bst = lgb.train(params, dtr, num_boost_round=700, valid_sets=[dva], valid_names=["va"],
                callbacks=[lgb.early_stopping(40), lgb.log_evaluation(25)])
bst.save_model(os.path.join(HERE, out_name + ".txt"))
imp = bst.feature_importance("gain"); order = np.argsort(-imp)
print("top features by gain:")
for i in order[:30]:
    print(f"   {PAIR_NAMES[i]:16s} {imp[i]/imp.sum():.3f}")
from sklearn.metrics import roc_auc_score
p = bst.predict(Xva, num_threads=6)
print("valid AUC", roc_auc_score(yva, p), "best iter", bst.best_iteration, "time", round(time.time() - t0, 1), flush=True)
print("done")
