import os, sys, time
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rs = np.random.RandomState(0)
X = rs.randn(225000, 37).astype(np.float32); T = (X[:, 3] + rs.randn(225000) > 1.5).astype(int)
for extra in (dict(num_threads=3), dict(num_threads=3, force_col_wise=True), dict(num_threads=3, force_row_wise=True), dict(num_threads=1)):
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, **extra)
    t = time.time(); lgb.train(P, lgb.Dataset(X, T), 300); print(extra, f"{time.time() - t:.1f}s", flush=True)
