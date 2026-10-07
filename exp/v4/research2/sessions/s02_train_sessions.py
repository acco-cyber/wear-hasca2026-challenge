"""training session ground truth: block switch points vs video discontinuities (consecutive-tile jumps)"""
from common import *

d = base(); y = d["oof_y"]
V = np.load(os.path.join(W, "data", "prep", "train_vid_mean768.npy")).astype(np.float32)
E = np.load(os.path.join(K7, "oof_emb.npy")).astype(np.float32)
# are V and E in the same row order?  (prep arrays are in train_meta order; stage arrays in OOF order)
m = pd.read_csv(os.path.join(W, "data", "prep", "train_meta.csv"))
names = sorted(m.session.unique())                      # rec index == sorted session name
key_m = {(s, t): i for i, (s, t) in enumerate(zip(m.session.values, m.t.values))}
perm = np.array([key_m[(names[r], t)] for r, t in zip(d["oof_rec"], d["t"])])   # OOF row -> meta row
np.save(os.path.join(OUT, "perm_oof2meta.npy"), perm)
Vo = V[perm]
c = (unitV := Vo / np.linalg.norm(Vo, axis=1, keepdims=True)) * (E / np.linalg.norm(E, axis=1, keepdims=True))
print("cos(train_vid_mean768[perm], oof_emb) median", np.median(c.sum(1)))
for r in np.unique(d["oof_rec"]):
    o = ordered(d, r); yy = y[o]; b = BLK[yy]
    U = unitV[o]; cs = (U[1:] * U[:-1]).sum(1)
    # block switch: positions where the block of consecutive non-null labels changes
    nz = np.flatnonzero(b > 0); sw = [(nz[k], nz[k + 1]) for k in range(len(nz) - 1) if b[nz[k]] != b[nz[k + 1]]]
    top = np.argsort(cs)[:4]
    win = 30
    # mean-cos between the 30 s before/after each candidate point (scene change score)
    def scene(p):
        a, bb = U[max(0, p - win):p + 1].mean(0), U[p + 1:p + 1 + win].mean(0)
        return float(a @ bb / np.linalg.norm(a) / np.linalg.norm(bb))
    sc = np.array([scene(p) for p in range(len(o) - 1)])
    tops = []
    for p in np.argsort(sc):
        if all(abs(p - q) > 60 for q in tops):
            tops.append(p)
        if len(tops) == 4:
            break
    print(f"rec {r:2d} {names[r]:9s} n {len(o)}  block switches {[(int(a), int(bb)) for a, bb in sw][:6]}  "
          f"min consecutive cos {[(int(p), round(float(cs[p]), 2)) for p in top]}  scene drops {[(int(p), round(float(sc[p]), 2)) for p in tops]}")
