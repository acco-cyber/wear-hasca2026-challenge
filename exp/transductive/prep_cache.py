"""Cache per-session inputs for refinement experiments: P (blend OOF of the simulated limb), y, raw feature blocks,
link graph, chains, baseline decoded labels. -> cache_<which>.pkl
python prep_cache.py eval|extra
"""
import sys, os, time, pickle
from tlib import *
from sklearn.metrics import f1_score

def main(which):
    S = load_structs(which); oof = load_oof_blend()
    imu = np.load(os.path.join(PREP, "train_imu.npy"), mmap_mode="r"); vp = np.load(os.path.join(PREP, "train_vid_pca.npy"), mmap_mode="r")
    v768 = np.load(os.path.join(PREP, "train_vid_mean768.npy"), mmap_mode="r")
    out = {}
    for s, st in S.items():
        t0 = time.time(); n = st["n"]; a, b = st["a"], st["b"]
        P = session_P(oof, st); y = st["y"]
        imu_w = np.asarray(imu[a:b], np.float32)[np.arange(n), st["limb"]]
        F = window_features(vp[a:b], v768[a:b], imu_w, st["limb"])
        g, chains = prep_decoder(st, n)
        lab, Pg, bias = decode(P, g, chains)
        f = f1_score(y, lab, average="macro")
        out[s] = dict(n=n, y=y, P=P, F=F, g=g, chains=chains, lab0=lab, Pg0=Pg, b0=bias, f0=f, limb=st["limb"])
        print(f"{s}: n={n} raw {f1_score(y, P.argmax(1), average='macro'):.4f} base {f:.4f} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(out, open(os.path.join(TD, f"cache_{which}.pkl"), "wb"))
    print("mean base", np.mean([v["f0"] for v in out.values()]).round(4))

if __name__ == "__main__":
    main(sys.argv[1])
