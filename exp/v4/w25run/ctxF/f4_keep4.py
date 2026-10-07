"""Write a keep4 copy whose stage-B base carries the chain-context experts:
   B2' = lsm(B2 + w * ctx_imu[key] + v * ctxB[W] + u * ctxQ[W])
OOF from cache/ctx_oof.npz, ctxB_oof_<fit>.npz, ctxQ_oof_<fit>.npz; with --test the same from the *_test files, otherwise
B2_TEST is left unchanged (placeholder, OOF evaluation only).  Everything else is the source fit's own (files copied).
  python f4_keep4.py --src <keep4> --out <keep4> --fit K7|K9 [--imu KEY:w] [--B W:v] [--Q W:u] [--test]"""
import os, sys, shutil, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctxlib import *


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fit", required=True)
    ap.add_argument("--imu", default=""); ap.add_argument("--B", default=""); ap.add_argument("--Q", default=""); ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = dict(np.load(os.path.join(a.src, "stage.npz"), allow_pickle=True))
    y = st["oof_y"].astype(np.int64); fold = st["oof_fold"].astype(np.int64)
    comps = []
    if a.imu:
        k, w = a.imu.rsplit(":", 1); comps.append(("ctx_oof.npz", "ctx_test.npz", k, float(w)))
    if a.B:
        k, w = a.B.split(":"); comps.append((f"ctxB_oof_{a.fit}.npz", f"ctxB_test_{a.fit}.npz", f"W{k}", float(w)))
    if a.Q:
        k, w = a.Q.split(":"); comps.append((f"ctxQ_oof_{a.fit}.npz", f"ctxQ_test_{a.fit}.npz", f"W{k}", float(w)))
    b0 = st["B2_OOF"].astype(np.float64); x = b0.copy()
    for fo, _, k, w in comps:
        x += w * np.load(os.path.join(CACHE, fo))[k].astype(np.float64)
    st["B2_OOF"] = lsm(x).astype(np.float32)
    msg = f"{comps}: B2 OOF tile F1 {macro_f1(y, b0.argmax(1)):.4f} -> {macro_f1(y, st['B2_OOF'].argmax(1)):.4f} | per fold " + " ".join(
        f"{macro_f1(y[fold == f], st['B2_OOF'].argmax(1)[fold == f]):.4f}" for f in range(5))
    if a.test:
        bt = st["B2_TEST"].astype(np.float64); xt = bt.copy()
        for _, ft, k, w in comps:
            zt = np.load(os.path.join(CACHE, ft)); assert (zt["ids"] == st["ids"]).all()
            xt += w * zt[k].astype(np.float64)
        st["B2_TEST"] = lsm(xt).astype(np.float32)
        msg += f"; test argmax changed {np.mean(st['B2_TEST'].argmax(1) != bt.argmax(1)):.4f} (OOF changed {np.mean(st['B2_OOF'].argmax(1) != b0.argmax(1)):.4f})"
    else:
        msg += "; B2_TEST unchanged (placeholder)"
    np.savez(os.path.join(a.out, "stage.npz"), **st)
    for f in ("links.npz", "oof_emb.npy", "test_emb.npy", "tile_scalars.npz", "dec_cache.npz", "link_logodds.npz"):
        if not os.path.exists(os.path.join(a.out, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    print(msg, "| wrote", a.out)


if __name__ == "__main__":
    main()
