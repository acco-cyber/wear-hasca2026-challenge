import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
sm = load_meta(); y, fold = sm["y"], sm["fold"]
win, S3b, Tb, LBb = base_parts()
te = fold == 0
for v, B in (("S3", S3b), ("T", Tb)):
    Q = np.load(os.path.join(HERE, "runs", "ctrl", f"{v}_f0.npy"))
    print(v, "max abs diff", float(np.abs(Q - B[te]).max()), "argmax agree", np.mean(Q.argmax(1) == B[te].argmax(1)),
          "F1 ctrl", round(macro_f1(y[te], Q.argmax(1)), 4), "F1 base", round(macro_f1(y[te], B[te].argmax(1)), 4))
