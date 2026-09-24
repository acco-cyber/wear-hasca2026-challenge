"""Mix old (work/sim_struct.pkl) and new links structs: chains from one, graph from the other / union; decoder variants.
python combo.py --new sim_struct_v2a.pkl"""
import argparse, time
from lk import *

def union_graph(g1, g2, w2=1.0):
    out = []
    for (i1, w1), (i2, ww2) in zip(g1, g2):
        out.append((np.concatenate([i1, i2]), np.concatenate([w1, w2 * ww2])))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--dec", default="new")
    a = ap.parse_args(); t0 = time.time()
    S0 = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); S1 = pickle.load(open(os.path.join(EXP, a.new), "rb")); oof = blend_oof(0.2)
    D = DEC_NEW if a.dec == "new" else DEC_OLD
    res = {}
    for s in EVAL:
        o, nw = S0[s], S1[s]; n = o["n"]; y = o["y"]; assert (o["limb"] == nw["limb"]).all()
        P = sim_P(oof, o)
        g_o = build_graph(o["cand"], o["lo"], n, k=10); g_n = build_graph(nw["cand"], nw["lo"], n, k=10)
        g_n5 = build_graph(nw["cand"], nw["lo"], n, k=5); g_n20 = build_graph(nw["cand"], nw["lo"], n, k=20)
        V = {
            "old": (o, g_o, -6.0), "new": (nw, g_n, -6.0),
            "chain_new_graph_old": (nw, g_o, -6.0), "chain_old_graph_new": (o, g_n, -6.0),
            "chain_new_graph_union": (nw, union_graph(g_o, g_n), -6.0), "chain_old_graph_union": (o, union_graph(g_o, g_n), -6.0),
            "new_k5": (nw, g_n5, -6.0), "new_k20": (nw, g_n20, -6.0), "new_thr-3": (nw, g_n, -3.0), "new_thr-9": (nw, g_n, -9.0),
        }
        for nm, (st, g, thr) in V.items():
            lab = decode(P, None, None, st["succ0"], st["sc"], st["Lm"], thr=thr, graph=g, **D)
            res.setdefault(nm, []).append(f1(y, lab))
        print(s, f"({time.time()-t0:.0f}s)", flush=True)
    for nm, v in res.items(): print(f"{nm:28s} mean {np.mean(v):.4f} | " + " ".join(f"{x:.4f}" for x in v))
