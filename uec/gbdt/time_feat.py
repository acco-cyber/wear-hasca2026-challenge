import time, sys
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\uec\gbdt")
from uecstub import tig

X = np.load(r"E:\Claude code\wear\data\test\test_inertial_data.npy", mmap_mode="r")
print(X.shape, X.dtype)
t = time.time()
for i in range(200):
    f = tig.make_feature_vector(np.asarray(X[i], np.float32), "ra", use_raw_features=False, smoothing_window=5,
                                sensor_embedding_mode="sensor")
print("per window ms", (time.time() - t) / 200 * 1000, "nfeat", len(f))
