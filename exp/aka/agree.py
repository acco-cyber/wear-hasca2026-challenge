"""python agree.py <test_probs.npy> [...]: argmax agreement with the akhyar2612/0-670 public submission."""
import sys
import numpy as np, pandas as pd
sub = pd.read_csv(r"E:\Claude code\wear\public_subs\akhyar2612__0-670\submission.csv"); a = sub.target_feature.to_numpy()
for p in sys.argv[1:]:
    T = np.load(p); q = T.argmax(1)
    print(f"{p}: agreement {np.mean(q == a):.4f}  null frac ours {np.mean(q == 0):.3f} akhyar {np.mean(a == 0):.3f}")
    if T.shape[1] == 19:
        Tb = T.copy(); Tb[:, 0] /= np.exp(0.75)
