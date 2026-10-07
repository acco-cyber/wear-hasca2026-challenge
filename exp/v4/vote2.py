"""Majority votes over the best decodes (ties -> first source), scored out-of-fold; writes the test CSV of each vote.
  python vote2.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1, write_sub, W

S = os.path.join(W, "subs")
K7 = np.load(os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4", "stage.npz"), allow_pickle=True)
K9 = np.load(os.path.join(W, "work", "v4", "wear-v4-big-tf-opt-s9", "keep4", "stage.npz"), allow_pickle=True)
y, fold = K7["oof_y"].astype(int), K7["oof_fold"].astype(int)
ld = lambda t: (np.load(os.path.join(S, f"sub_v4c_{t}_labo.npy")), np.load(os.path.join(S, f"sub_v4c_{t}_labt.npy")))
SRC = {"b4wa": ld("b4wa"), "b4wb": ld("b4wb"), "clog": ld("b4wa_clog"), "b5ab": ld("b5ab_clog"),
       "s7": (K7["ref_oof"], K7["ref_test"]), "s9": (K9["ref_oof"], K9["ref_test"])}


def vote(names, split):
    L = np.stack([SRC[n][split].astype(int) for n in names]); n = L.shape[1]
    cnt = np.zeros((n, 19)); np.add.at(cnt, (np.tile(np.arange(n), len(names)), L.ravel()), 1.0)
    cnt[np.arange(n), L[0]] += 0.5
    return cnt.argmax(1)


for k, (o, _) in SRC.items():
    print(f"{k}: OOF {macro_f1(y, o.astype(int)):.4f}")
for names in (["b4wa", "s7", "b4wb"], ["b5ab", "b4wa", "b4wb", "clog", "s7"], ["b5ab", "b4wa", "s7", "s9", "b4wb"]):
    lab = vote(names, 0); pf = [macro_f1(y[fold == k], lab[fold == k]) for k in range(5)]
    out = os.path.join(S, f"sub_vote2_{'_'.join(names)}.csv"); write_sub(K7["ids"], vote(names, 1), out)
    print(f"vote {'+'.join(names)}: OOF {macro_f1(y, lab):.4f} | per fold " + " ".join(f"{x:.4f}" for x in pf) + f" -> {os.path.basename(out)}")
