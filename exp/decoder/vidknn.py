"""Within-subject video kNN edges."""
import numpy as np
from common import *

_V = {}
def train_vid():
    if "v" not in _V: _V["v"] = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
    return _V["v"]

def mp_desc(V):
    V = np.asarray(V, np.float32); Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)
    m = Vn.mean(1); return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

def knn_edges(D, k=10, exclude_links=None):
    S = D @ D.T; np.fill_diagonal(S, -1); n = len(D)
    nb = np.argpartition(-S, k, axis=1)[:, :k]; sim = S[np.arange(n)[:, None], nb]
    R = np.repeat(np.arange(n), k); C = nb.ravel(); w = sim.ravel()
    return R, C, w

if __name__ == "__main__":
    S = load_structs(("eval", "extra"))
    for s in EVAL + EXTRA[:2]:
        st = S[s]; y = st["y"]; D = mp_desc(train_vid()[st["a"]:st["b"]])
        R, C, Lo = graph_edges(st["cand"], st["lo"], k=10)
        print(s, "link top10 same-label %.3f" % (y[R] == y[C]).mean(), end=" | ")
        for k in (5, 10, 20):
            r, c, w = knn_edges(D, k); print(f"vid k{k} same %.3f" % (y[r] == y[c]).mean(), end=" | ")
        # time distance of video neighbours
        r, c, w = knn_edges(D, 10); print("vid k10 |dt|<=3: %.3f" % (np.abs(r - c) <= 3).mean(), "link |dt|<=3: %.3f" % (np.abs(R - C) <= 3).mean())
