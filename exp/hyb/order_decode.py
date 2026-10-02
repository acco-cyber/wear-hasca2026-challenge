"""Decoder given a (possibly imperfect) reconstructed time order: our mrf4 chain Viterbi + count calibration run on
chains built from successor links. CV on the Hanbat OOF rows where the true order is known.
  python order_decode.py cv <their_oof_P.npy> [--break r] [--wrong r] [--variant mrf4] [--mix w]
    --break r : drop a fraction r of true links (fragments the chains)
    --wrong r : replace a fraction r of links by a random same-recording window (wrong successor)
    --mix w   : blend P with the graph P before decoding (w = weight of the decoded one-hot in a final vote; 0 = off)
  used by test code via decode_with_succ(P, succ, score, sbj)"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\decoder"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder import decode_subject, VARIANTS
from hanbat_stack import finish, macro_f1, KEEP, TRAIN_SETS, N_CLS


def make_struct(succ_local, score_local):
    n = len(succ_local)
    cand = np.full((n, 2), -1, np.int64); lo = np.full((n, 2), -50.0, np.float32)
    cand[:, 0] = succ_local; lo[:, 0] = np.where(succ_local >= 0, score_local, -50.0)
    pred = np.full(n, -1, np.int64); m = succ_local >= 0; pred[succ_local[m]] = np.flatnonzero(m)
    cand[:, 1] = pred; lo[:, 1] = np.where(pred >= 0, score_local[np.maximum(pred, 0)], -50.0)
    return dict(cand=cand, lo=lo, succ0=succ_local.astype(np.int64), sc=np.where(m, score_local, -50.0).astype(np.float32),
                Lm=np.zeros((1, 1), np.float32), n=n)


def decode_group(P, succ_local, score_local, variant="mrf4"):
    st = make_struct(succ_local, score_local)
    return decode_subject(P, st, VARIANTS[variant], None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("P")
    ap.add_argument("--break_", "--break", dest="brk", type=float, default=0.0); ap.add_argument("--wrong", type=float, default=0.0)
    ap.add_argument("--variant", default="mrf4"); ap.add_argument("--score", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    P = np.load(a.P).astype(np.float64); P /= P.sum(1, keepdims=True)
    sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
    y, rec, st_, sbj, fold = sm["y"], sm["rec"], sm["start"], sm["sbj"], sm["fold"]
    base = finish(P, dict(sbj=sbj, sets=TRAIN_SETS)); print(f"graph baseline F1 {macro_f1(y, base):.4f}")
    rng = np.random.default_rng(a.seed); out = base.copy()
    for r in np.unique(rec):
        ii = np.flatnonzero(rec == r); ii = ii[np.argsort(st_[ii])]; n = len(ii)
        succ = np.r_[np.arange(1, n), -1].astype(np.int64)
        gap = np.diff(st_[ii]) != 50                     # missing seconds -> no link
        succ[:-1][gap] = -1
        if a.brk > 0:
            succ[rng.random(n) < a.brk] = -1
        if a.wrong > 0:
            w = (rng.random(n) < a.wrong) & (succ >= 0); succ[w] = rng.integers(0, n, w.sum())
            # keep 1:1 (drop duplicates targets) and no self-links
            succ[succ == np.arange(n)] = -1
            _, first = np.unique(np.where(succ >= 0, succ, -np.arange(1, n + 1)), return_index=True)
            keep = np.zeros(n, bool); keep[first] = True; succ[~keep] = -1
        score = np.full(n, a.score, np.float32)
        out[ii] = decode_group(P[ii], succ, score, a.variant)
    f = macro_f1(y, out)
    print(f"order decode ({a.variant}, break {a.brk}, wrong {a.wrong}): F1 {f:.4f} ({f - macro_f1(y, base):+.4f})  per fold "
          + " ".join(f"{macro_f1(y[fold == k], out[fold == k]):.4f}" for k in range(5)))


if __name__ == "__main__":
    main()
