import os, sys, time
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *
subs = list(range(22))
X = np.concatenate([np.load(os.path.join(CACHE, f"imuf_s{s}.npz"))["F"].reshape(-1, 107) for s in subs])
y = np.random.default_rng(0).integers(0, 19, len(X))
m = np.random.default_rng(1).random(len(X)) < 0.8
P = dict(objective="multiclass", num_class=N_CLS, learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, max_bin=63, verbose=-1, num_threads=2)
t0 = time.time(); ds = lgb.Dataset(X[m], y[m]); ds.construct(); print("construct", time.time() - t0)
t0 = time.time(); b = lgb.train(P, ds, 10); print("10 rounds", time.time() - t0)
