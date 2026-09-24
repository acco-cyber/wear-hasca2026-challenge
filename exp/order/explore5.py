"""Per true set (contiguous segment) label distribution of e7 within families (push-ups, sit-ups, lunges)."""
import pickle
import numpy as np
R = pickle.load(open(r"E:\Claude code\wear\exp\pl\labels_v3bv1f.pkl", "rb"))
for s in R:
    y = R[s]["y"]; lab = R[s]["lab"]; n = len(y)
    b = np.r_[0, np.where(np.diff(y) != 0)[0] + 1, n]
    out = []
    for i in range(len(b) - 1):
        l = int(y[b[i]])
        if l not in (11, 12, 13, 14, 16, 17): continue
        seg = lab[b[i]:b[i + 1]]; bc = np.bincount(seg, minlength=19)
        top = [(int(c), int(bc[c])) for c in np.argsort(-bc)[:3] if bc[c] > 0]
        out.append(f"t{b[i]}:{l}[{b[i+1]-b[i]}]->" + ",".join(f"{c}:{k}" for c, k in top))
    print(s, " | ".join(out))
