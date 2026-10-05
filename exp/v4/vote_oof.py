"""Majority vote over several decodes' final labels, scored out-of-fold; ties go to the first source.
  python vote_oof.py   (sources are listed below)"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1, write_sub, W

S = os.path.join(W, "subs")
K7 = np.load(os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4", "stage.npz"), allow_pickle=True)
K9 = np.load(os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4", "stage.npz"), allow_pickle=True)
y, fold = K7["oof_y"].astype(int), K7["oof_fold"].astype(int)
SRC = {
    "b4": (np.load(os.path.join(S, "sub_v4c_b4_s7s9e7e9_labo.npy")), np.load(os.path.join(S, "sub_v4c_b4_s7s9e7e9_labt.npy"))),
    "b2": (np.load(os.path.join(S, "sub_v4c_b2_s7s9_labo.npy")), np.load(os.path.join(S, "sub_v4c_b2_s7s9_labt.npy"))),
    "s7": (K7["ref_oof"], K7["ref_test"]),
    "s9": (K9["ref_oof"], K9["ref_test"]),
    "e7": (np.load(os.path.join(S, "sub_v4l_e7_f1_whs_oh_labo.npy")), np.load(os.path.join(S, "sub_v4l_e7_f1_whs_oh_labt.npy"))),
}
for k, (o, t) in SRC.items():
    print(f"{k}: OOF {macro_f1(y, o.astype(int)):.4f}")


def vote(names, split):
    L = np.stack([SRC[n][split].astype(int) for n in names])
    cnt = np.zeros((L.shape[1], 19)); np.add.at(cnt, (np.arange(L.shape[1])[None].repeat(len(names), 0), L), 1.0)
    cnt[np.arange(L.shape[1]), L[0]] += 0.5                  # tie-break: first source
    return cnt.argmax(1)


best = None
for r in (3, 5):
    for names in itertools.combinations(SRC, r):
        if names[0] != "b4":
            continue
        lab = vote(names, 0); f = macro_f1(y, lab)
        pf = [macro_f1(y[fold == k], lab[fold == k]) for k in range(5)]
        print(f"vote {'+'.join(names)}: OOF {f:.4f} | per fold " + " ".join(f"{x:.4f}" for x in pf))
        if best is None or f > best[0]:
            best = (f, names)
print("best vote:", best)
if len(sys.argv) > 1:
    names = sys.argv[1].split("+"); lab = vote(names, 1)
    out = os.path.join(S, f"sub_vote_{'_'.join(names)}.csv"); write_sub(K7["ids"], lab, out); print("wrote", out)
