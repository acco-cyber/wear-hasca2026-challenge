import numpy as np
z = np.load(r"E:\Claude code\wear\work\w25\w25.npz"); s, l = z["sbj"], z["limb"]
ch = np.flatnonzero(np.diff(l) != 0); print("limb change points", ch[:20], len(ch))
blocks = [np.flatnonzero(l == L) for L in range(4)]
print("block ranges", [(b.min(), b.max(), len(b)) for b in blocks])
for A in range(4):
    for B in range(A + 1, 4):
        print(A, B, "same sbj sequence", np.mean(s[blocks[A]] == s[blocks[B]]))
print("sbj runs in block 0 (first 40)", s[blocks[0]][:40])
