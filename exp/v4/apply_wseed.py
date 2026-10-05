"""Put a new window blend (from a kaggle/v4_fork/patch_wseed.py job) into a copy of a fit's keep4: the stage-B base
B2 = log-softmax(0.2 * window + 0.5 * S3 + 0.3 * adapted T) is rebuilt with it; everything else is the fit's own.
  python apply_wseed.py <keep4 dir> <window_logp.npz> <out keep4 dir>"""
import os, sys, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_local import macro_f1


def tab_blend(lp, parts):
    x = (1 - sum(w for _, w in parts)) * lp
    for q, w in parts:
        x = x + w * q
    x = x - x.max(1, keepdims=True)
    return (x - np.log(np.exp(x).sum(1, keepdims=True))).astype(np.float32)


def main():
    src, wl, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    st = dict(np.load(os.path.join(src, "stage.npz"), allow_pickle=True)); z = np.load(wl)
    lo, lt = z["oof_logp"].astype(np.float64), z["test_logp"].astype(np.float64)
    chk = tab_blend(st["oof_logp"].astype(np.float64), [(st["tab_S3_oof"].astype(np.float64), 0.5), (st["TA_OOF"].astype(np.float64), 0.3)])
    assert np.abs(chk - st["B2_OOF"]).max() < 1e-3
    y = st["oof_y"].astype(int)
    st["B2_OOF"] = tab_blend(lo, [(st["tab_S3_oof"].astype(np.float64), 0.5), (st["TA_OOF"].astype(np.float64), 0.3)])
    st["B2_TEST"] = tab_blend(lt, [(st["tab_S3_test"].astype(np.float64), 0.5), (st["TA_TEST"].astype(np.float64), 0.3)])
    print(f"window tile F1 {macro_f1(y, chk.argmax(1) * 0 + st['oof_logp'].argmax(1)):.4f} -> {macro_f1(y, lo.argmax(1)):.4f}; "
          f"B2 tile F1 {macro_f1(y, chk.argmax(1)):.4f} -> {macro_f1(y, st['B2_OOF'].argmax(1)):.4f}")
    st["oof_logp"], st["test_logp"] = lo.astype(np.float32), lt.astype(np.float32)
    np.savez(os.path.join(out, "stage.npz"), **st)
    for f in ("links.npz", "oof_emb.npy", "test_emb.npy", "tile_scalars.npz", "dec_cache.npz"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    print("wrote", out)


if __name__ == "__main__":
    main()
