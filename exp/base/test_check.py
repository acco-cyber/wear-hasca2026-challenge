"""Sanity check that a model's test probs behave like its OOF (train/test feature-pipeline consistency)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
def mdir(m): return os.path.join(WORK, m[2:]) if m.startswith("w:") else os.path.join(EXP, m)
ref = sys.argv[1]; others = sys.argv[2].split(",")
Ro = np.load(os.path.join(mdir(ref), "oof.npy")); Rt = np.load(os.path.join(mdir(ref), "test.npy"))
tm = pd.read_csv(os.path.join(DATA, "test", "test_meta_data.csv"))
for m in others:
    Oo = np.load(os.path.join(mdir(m), "oof.npy")); Ot = np.load(os.path.join(mdir(m), "test.npy"))
    v = ~np.isnan(Ro[:, :, 0]) & ~np.isnan(Oo[:, :, 0])
    agree_oof = (Ro[v].argmax(1) == Oo[v].argmax(1)).mean(); agree_te = (Rt.argmax(1) == Ot.argmax(1)).mean()
    print(f"{m} vs {ref}: argmax agreement OOF {agree_oof:.3f} test {agree_te:.3f} | P(null) OOF {np.nanmean(Oo[v][:, 0]):.3f} test {Ot[:, 0].mean():.3f} | "
          f"argmax-null OOF {np.mean(Oo[v].argmax(1)==0):.3f} test {np.mean(Ot.argmax(1)==0):.3f} | maxp OOF {Oo[v].max(1).mean():.3f} test {Ot.max(1).mean():.3f}")
    for s in sorted(tm.sbj_id.unique()):
        mm = tm.sbj_id.to_numpy() == s; c = np.bincount(Ot[mm].argmax(1), minlength=NC)
        print(f"   sbj {s}: n={mm.sum()} argmax-null {c[0]/mm.sum():.3f} classes>=20: {(c[1:]>=20).sum()}")
