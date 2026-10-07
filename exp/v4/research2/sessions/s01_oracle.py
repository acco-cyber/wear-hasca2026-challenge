"""(1) error decomposition of the current decodes by block; (2) oracle upper bound of a perfect session/block mask"""
from common import *

d = base(); y = d["oof_y"]; tb = true_block(d)
S = os.path.join(W, "subs")
runs = {"K7_ref": (d["QB_OOF"], d["ref_oof"]),
        "b4wa": (np.load(os.path.join(S, "sub_v4c_b4wa_Qo.npy")), np.load(os.path.join(S, "sub_v4c_b4wa_labo.npy")).astype(int))}
for nm, (Q, lab) in runs.items():
    err = lab != y
    by, bp = BLK[y], BLK[lab]
    cross = err & (by > 0) & (bp > 0) & (by != bp)
    within = err & (by > 0) & (bp > 0) & (by == bp)
    n2a = err & (by == 0); a2n = err & (bp == 0)
    print(f"{nm}: F1 {f1(y, lab):.4f}  errors {err.sum()}  cross-block {cross.sum()}  within-block {within.sum()}  null->act {n2a.sum()}  act->null {a2n.sum()}")
    # predicted activity whose block disagrees with the (oracle) session block of the tile
    wrongblk = (bp > 0) & (bp != tb)
    print(f"   predicted activity in the wrong session block: {wrongblk.sum()} tiles (of which truly null {(wrongblk & (by == 0)).sum()})")
    for mode in ("label", "Q"):
        Qm = Q.copy().astype(np.float64)
        for b in (1, 2):
            forb = np.flatnonzero((BLK > 0) & (BLK != b))
            Qm[np.ix_(tb == b, forb)] = 0
        if mode == "Q":
            new = Qm.argmax(1)
        else:
            new = lab.copy(); new[wrongblk] = Qm[wrongblk].argmax(1)
        print(f"   oracle block mask ({mode}): F1 {f1(y, new):.4f}  ({f1(y, new) - f1(y, lab):+.4f})  per fold "
              + " ".join(f"{f1(y[d['oof_fold'] == k], new[d['oof_fold'] == k]):.4f}" for k in range(5)))
    # per-recording view: which recordings carry the cross-block errors
    rows = []
    for r in np.unique(d["oof_rec"]):
        ii = d["oof_rec"] == r
        rows.append((r, int(d["oof_sbj"][ii][0]), int((cross & ii).sum()), int((wrongblk & ii).sum()), int(ii.sum())))
    print("   rec sbj cross wrongblk n:", rows)
# confusion pairs across blocks
lab = runs["b4wa"][1]; err = (lab != y) & (BLK[y] > 0) & (BLK[lab] > 0) & (BLK[y] != BLK[lab])
pairs = pd.Series(list(zip(y[err], lab[err]))).value_counts().head(15)
print("top cross-block confusions (true, pred):"); print(pairs)
