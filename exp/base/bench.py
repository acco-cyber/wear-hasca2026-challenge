import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
from feats_v3 import train_blocks
IM, VB = train_blocks(); m = meta(); y = m.y.to_numpy()
rng = np.random.RandomState(0); s = rng.choice(len(m), 30000, replace=False); l = rng.randint(0, 4, len(s))
X = np.concatenate([IM[s, l], np.eye(4, dtype=np.float32)[l], VB[s]], 1); yy = y[s]
for th in [int(x) for x in sys.argv[1].split(",")]:
    t = time.time(); d = lgb.Dataset(X, yy, params=dict(max_bin=63, verbose=-1)); d.construct(); tc = time.time() - t
    p = dict(objective="multiclass", num_class=19, learning_rate=0.1, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.25,
             bagging_fraction=0.7, bagging_freq=1, max_bin=63, num_threads=th, verbose=-1, seed=0)
    t = time.time(); lgb.train(p, d, num_boost_round=10); print(f"threads {th}: construct {tc:.1f}s, 10 iters {time.time()-t:.1f}s", flush=True)
