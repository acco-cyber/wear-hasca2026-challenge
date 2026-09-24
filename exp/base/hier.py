"""Combine a flat 19-class model with a family (8) or binary (null vs activity) model.
Hierarchical re-mass: P_h(c) = P_aux(g(c)) * P_flat(c) / P_flat(g(c)); final = norm(exp((1-w) log P_flat + w log P_h)).
python hier.py --flat <dir> --aux <dir> --kind family|binary --out <name> [--w 0.3] [--sim]"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def remass(Pf, Pa, kind):
    G = FAMILY if kind == "family" else (np.arange(NC) > 0).astype(int)
    K = G.max() + 1
    Pg = np.stack([Pf[..., G == g].sum(-1) for g in range(K)], -1)          # flat mass per group
    Ph = Pa[..., G] * Pf / np.clip(Pg[..., G], 1e-9, None)
    return Ph / np.nansum(Ph, -1, keepdims=True)

def combine(Pf, Pa, kind, w):
    Ph = remass(Pf, Pa, kind); return blend([Pf, Ph], [1 - w, w])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--flat", required=True); ap.add_argument("--aux", required=True)
    ap.add_argument("--kind", default="binary"); ap.add_argument("--out", required=True); ap.add_argument("--w", type=float, default=-1)
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    fp = lambda p: p if os.path.isabs(p) else os.path.join(r"E:\Claude code\wear", p)
    Of = np.load(os.path.join(fp(a.flat), "oof.npy")); Oa = np.load(os.path.join(fp(a.aux), "oof.npy"))
    Tf = np.load(os.path.join(fp(a.flat), "test.npy")); Ta = np.load(os.path.join(fp(a.aux), "test.npy"))
    eval_single(Of, name="flat")
    best = (-1, None)
    for w in ([a.w] if a.w >= 0 else [0.2, 0.3, 0.5, 0.7, 1.0]):
        f = eval_single(combine(Of, Oa, a.kind, w), name=f"w={w}")
        if f > best[0]: best = (f, w)
    w = best[1]; od = os.path.join(EXP, a.out); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), combine(Of, Oa, a.kind, w)); np.save(os.path.join(od, "test.npy"), combine(Tf, Ta, a.kind, w))
    res = dict(flat=a.flat, aux=a.aux, kind=a.kind, w=w, oof_f1=best[0])
    if a.sim:
        s, _ = run_sim(os.path.join(od, "oof.npy"), a.out); res["sim"] = s; print("SIM raw", s.get("raw"), "chain", s.get("g_chain_cal_ps0.8"))
    json.dump(res, open(os.path.join(od, "res.json"), "w"), indent=1); print(res)

if __name__ == "__main__":
    main()
