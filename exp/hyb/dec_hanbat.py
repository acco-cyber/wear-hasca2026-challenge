"""Our decoder (mrf4 + video-kNN refinement, refine_ns.py) on top of the Hanbat pipeline's test arrays, and the
link-swap variant (their L2 successor links replace our chain links inside our decoder).

python dec_hanbat.py export                 -> work/hanbat/hb_{win,tab,graph,S3,T}.npy (12234,19) probabilities,
                                               work/test_structure_hbL2.pkl (our struct with their L2 links)
python dec_hanbat.py run <name> "<extra spec>" [--votes] [--tags fusion_v1:0.2] [--struct hbL2] [--w 6]
    extra spec = comma list of prob_file:weight where prob_file is a short name (hb_tab, uec_ens, v3b_full, aka_full, ...) or a path
"""
import os, sys, argparse, pickle, subprocess
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"; WORK = os.path.join(W, "work"); HB = os.path.join(WORK, "hanbat"); KEEP = os.path.join(HB, "keep")
SUBS = os.path.join(W, "subs")
VOTES = ",".join([os.path.join(W, "acco", "download", "sub_d7.csv") + ":1.0",
                  os.path.join(W, "public_subs", "nomannic19__ts-emb-3wdc-temporal-fusion-ensemble", "submission.csv") + ":0.5",
                  os.path.join(W, "acco", "download", "sub_d1.csv") + ":0.4",
                  os.path.join(W, "public_subs", "abhinavm2811__3rd-wear-dataset-challenge-hasca-2026", "submission.csv") + ":0.4"])
SHORT = {"hb_win": os.path.join(HB, "hb_win.npy"), "hb_tab": os.path.join(HB, "hb_tab.npy"), "hb_graph": os.path.join(HB, "hb_graph.npy"),
         "hb_S3": os.path.join(HB, "hb_S3.npy"), "hb_T": os.path.join(HB, "hb_T.npy"),
         "uec_ens": os.path.join(WORK, "uec_ens.npy"), "v3b_full": os.path.join(W, "exp", "full", "v3b_full", "test.npy"),
         "v1_full": os.path.join(W, "exp", "full", "v1_full", "test.npy"), "aka_full": os.path.join(W, "exp", "aka", "aka_full", "test.npy"),
         "fusion_full": os.path.join(WORK, "fusion_full", "test.npy")}
E39 = "uec_ens:1.0,v3b_full:0.4,aka_full:0.3"


def export():
    os.makedirs(HB, exist_ok=True)
    ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True); assert (ids == np.arange(len(ids))).all()
    bl = np.load(os.path.join(KEEP, "blend.npz"))
    def sp(x):
        P = np.exp(x.astype(np.float64)); return (P / P.sum(1, keepdims=True)).astype(np.float32)
    np.save(SHORT["hb_win"], sp(bl["test_logp"])); np.save(SHORT["hb_S3"], sp(bl["tab_S3"])); np.save(SHORT["hb_T"], sp(bl["tab_T"]))
    np.save(SHORT["hb_tab"], sp(np.load(os.path.join(KEEP, "test_logp_b.npy"))))
    P = np.load(os.path.join(KEEP, "P_test.npy")).astype(np.float32); np.save(SHORT["hb_graph"], P / P.sum(1, keepdims=True))
    ref = pd.read_csv(os.path.join(SUBS, "sub_e44_vote9.csv")).sort_values("id").target_feature.to_numpy()
    from sklearn.metrics import f1_score
    for k in ("hb_win", "hb_S3", "hb_T", "hb_tab", "hb_graph"):
        Q = np.load(SHORT[k]); a = Q.argmax(1)
        print(f"{k:9s} agree_e44 {np.mean(a == ref):.4f} F1_vs_e44 {f1_score(ref, a, average='macro'):.4f} null {np.mean(a == 0):.3f}")
    # link-swap structure: our per-subject struct with succ0/sc/Lm replaced by their L2 links (quantile-normalised scores)
    l2 = np.load(os.path.join(KEEP, "links_L2_test.npz")); succ_g, sc_g = l2["succ"].astype(np.int64), l2["score_qn"].astype(np.float32)
    struct = pickle.load(open(os.path.join(WORK, "test_structure.pkl"), "rb")); out = {}
    for s, st in struct.items():
        idx = np.asarray(st["idx"]); n = len(idx); pos = {int(g): i for i, g in enumerate(idx)}
        succ = np.array([pos.get(int(succ_g[g]), -1) if succ_g[g] >= 0 else -1 for g in idx], np.int64)
        sc = np.where(succ >= 0, sc_g[idx], -50.0).astype(np.float32)
        Lm = np.full((n, n), -50.0, np.float32); m = succ >= 0; Lm[np.flatnonzero(m), succ[m]] = sc[m]
        o = dict(st); o["succ0"] = succ; o["sc"] = sc; o["Lm"] = Lm; out[s] = o
        ours = np.asarray(st["succ0"]); same = np.mean(ours[m] == succ[m])
        print(f"sbj {s}: n={n} their linked {m.mean():.3f}, agree with our succ0 on linked rows {same:.3f}")
    pickle.dump(out, open(os.path.join(WORK, "test_structure_hbL2.pkl"), "wb")); print("saved test_structure_hbL2.pkl")


def run(name, extra, votes, tags, struct, w):
    py = sys.executable; env = dict(os.environ, PYTHONPATH=os.path.join(W, "shim"))
    spec = ",".join(f"{SHORT.get(p, p)}:{wt}" for p, wt in (x.rsplit(":", 1) for x in extra.split(",")))
    probs = os.path.join(WORK, f"probs_{name}.npy")
    cmd = [py, os.path.join(W, "src", "make_probs.py"), probs, "--tags", tags, "--extra", spec]
    if votes:
        cmd += ["--votes", VOTES]
    subprocess.run(cmd, check=True, env=env)
    out = os.path.join(SUBS, f"sub_{name}.csv")
    cmd = [py, os.path.join(W, "exp", "transductive", "refine_ns.py"), "--probs", probs, "--out", out, "--w", str(w)]
    if struct:
        cmd += ["--struct", os.path.join(WORK, f"test_structure_{struct}.pkl")]
    subprocess.run(cmd, check=True, env=env, cwd=os.path.join(W, "exp", "transductive"))
    lab = pd.read_csv(out).sort_values("id").target_feature.to_numpy()
    from sklearn.metrics import f1_score
    for ref_name in ("sub_e44_vote9.csv", "sub_uec3_u3_uec_primary.csv"):
        ref = pd.read_csv(os.path.join(SUBS, ref_name)).sort_values("id").target_feature.to_numpy()
        print(f"==> {name} vs {ref_name}: agree {np.mean(lab == ref):.4f} F1 {f1_score(ref, lab, average='macro'):.4f}")
    hb = pd.read_csv(os.path.join(W, "public_src", "woominyo", "out", "submission.csv")).sort_values("id").target_feature.to_numpy()
    print(f"==> {name} vs public hanbat 0.890: agree {np.mean(lab == hb):.4f}; null {np.mean(lab == 0):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("name", nargs="?"); ap.add_argument("extra", nargs="?")
    ap.add_argument("--votes", action="store_true"); ap.add_argument("--tags", default="none"); ap.add_argument("--struct", default="")
    ap.add_argument("--w", type=float, default=6.0)
    a = ap.parse_args()
    if a.mode == "export":
        export()
    else:
        run(a.name, a.extra, a.votes, a.tags, a.struct, a.w)
