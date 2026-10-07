import numpy as np
z = np.load(r"E:\Claude code\wear\work\w25\w25.npz"); l = z["limb"]
t = np.load(r"E:\Claude code\wear\exp\v4\w25run\ctxF\cache\test_twins.npz")["twin"]
start = {L: np.flatnonzero(l == L).min() for L in range(4)}
pos = np.array([r - start[l[r]] for r in t])
print("distinct positions", len(np.unique(pos)), "of", len(pos))
print("corr(id, pos)", np.corrcoef(np.arange(len(pos)), pos)[0, 1])
print("first 20 pos", pos[:20])
