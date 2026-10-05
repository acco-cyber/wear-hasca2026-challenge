"""Bout-level swap check: per subject, every predicted activity group (tiles predicted class a) is scored against every
class b with the summed per-tile evidence log p(b) (window blend or stage-B blend); a one-to-one assignment of groups to
classes (null fixed) relabels groups whose evidence prefers another class by a margin. OOF report only.
  python swapfix.py <Qo.npy> <stage.npz> [--src B2_OOF|oof_logp] [--margin 0]"""
import sys, argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb")
from hanbat_stack import macro_f1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("Qo"); ap.add_argument("stage"); ap.add_argument("--src", default="B2_OOF")
    ap.add_argument("--margin", type=float, default=0.0)
    a = ap.parse_args()
    st = np.load(a.stage); y = st["oof_y"].astype(int); sbj = st["oof_sbj"].astype(int); fold = st["oof_fold"].astype(int)
    lab = np.load(a.Qo).argmax(1); L = st[a.src].astype(np.float64)
    new = lab.copy(); changed = []
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s)
        M = np.zeros((18, 18))
        for ga in range(1, 19):
            g = ii[lab[ii] == ga]
            if len(g):
                M[ga - 1] = L[g][:, 1:].sum(0)
        r, c = linear_sum_assignment(-M)
        for ga, cb in zip(r, c):
            if ga != cb and M[ga, cb] - M[ga, ga] > a.margin * max(1, (lab[ii] == ga + 1).sum()):
                g = ii[lab[ii] == ga + 1]; new[g] = cb + 1
                changed.append((int(s), ga + 1, cb + 1, len(g), float((y[g] == ga + 1).mean()), float((y[g] == cb + 1).mean())))
    print(f"{a.src} margin {a.margin}: groups relabelled {len(changed)}; OOF F1 {macro_f1(y, lab):.4f} -> {macro_f1(y, new):.4f}")
    for ch in changed:
        print(f"  sbj {ch[0]}: group {ch[1]} -> {ch[2]} ({ch[3]} tiles; truly old {ch[4]:.2f}, truly new {ch[5]:.2f})")


if __name__ == "__main__":
    main()
