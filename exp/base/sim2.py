"""Compact sim evaluation for several OOF files in one process (loads sim_struct once).
old = sim_decode.py g_chain_cal_ps0.8 (ns 1.0, band 60-160); new = lead's decoder (null_scale before graph smoothing, band 80-250, ps 0.8).
python sim2.py path1[,path2,...] [--ns 0.5] [--nsgrid 0.3,0.5,0.7,1.0]"""
import os, sys, pickle, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
sys.path.insert(0, r"E:\Claude code\wear\src")
from chain import cut, chains_from_succ
from decode import calibrate_counts, build_graph, graph_smooth

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths"); ap.add_argument("--nsgrid", default="0.5"); ap.add_argument("--old", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    S = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb"))
    G = {s: (build_graph(st["cand"], st["lo"], st["n"], k=10), chains_from_succ(cut(st["succ0"], st["sc"], st["Lm"], -6.0))) for s, st in S.items()}
    print(f"struct loaded {time.time()-t0:.0f}s", flush=True)
    nsg = [float(x) for x in a.nsgrid.split(",")]
    for p in a.paths.split(","):
        p = os.path.join(WORK, p[2:], "oof.npy") if p.startswith("w:") else (p if os.path.isabs(p) else os.path.join(EXP, p, "oof.npy"))
        oof = np.load(p); res = {}
        for s, st in S.items():
            n = st["n"]; y = st["y"]; g, chains = G[s]
            P = oof[st["a"]:st["b"]][np.arange(n), st["limb"]].astype(np.float64); ok = ~np.isnan(P[:, 0]); P = np.where(ok[:, None], P, 1.0 / 19)
            res.setdefault("raw", []).append(f1_score(y, P.argmax(1), average="macro"))
            if a.old:
                Pg = graph_smooth(P, g, alpha=0.5, iters=5); lab, _ = calibrate_counts(Pg, chains, lo=60, hi=160, p_stay=0.8)
                res.setdefault("old", []).append(f1_score(y, lab, average="macro"))
            for ns in nsg:
                Pn = P.copy(); Pn[:, 0] *= ns; Pn /= Pn.sum(1, keepdims=True); Pg = graph_smooth(Pn, g, alpha=0.5, iters=5)
                lab, _ = calibrate_counts(Pg, chains, lo=80, hi=250, p_stay=0.8)
                res.setdefault(f"new_ns{ns}", []).append(f1_score(y, lab, average="macro"))
        line = " | ".join(f"{k} {np.mean(v):.4f} [{' '.join(f'{x:.3f}' for x in v)}]" for k, v in res.items())
        print(f"SIM2 {p}: {line} ({time.time()-t0:.0f}s)", flush=True)
        with open(os.path.join(EXP, "sim2_log.txt"), "a") as f: f.write(f"{p}\t" + "\t".join(f"{k}={np.mean(v):.4f}" for k, v in res.items()) + "\n")

if __name__ == "__main__":
    main()
