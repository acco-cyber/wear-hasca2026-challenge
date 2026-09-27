"""Assemble the UEC-dx2 ensemble from whichever member test-probability files exist (UEC beam weights, arithmetic
weighted average as in their ensemble.py, weights renormalised over available members), then build candidate
probability files with our recipe and decode them.
python uec_assemble.py [--decode]  -> work/uec_ens.npy, work/uec_imu.npy, and subs/sub_u*.csv when --decode"""
import os, sys, argparse, subprocess
import numpy as np, pandas as pd
W = r"E:\Claude code\wear"; WORK = os.path.join(W, "work"); SUBS = os.path.join(W, "subs")
MEMBERS = {  # name: (path, UEC beam weight, inertial-only?)
    "lgbm":    (os.path.join(W, "uec", "gbdt", "full", "test_prob.npy"), 0.272, True),
    "cnn8vid": (os.path.join(W, "work", "uec_k1", "test_cnn8.npy"), 0.221, False),
    "xcep":    (os.path.join(W, "work", "uec_k2", "test_xcep.npy"), 0.203, True),
    "vmlp":    (os.path.join(W, "work", "uec_k2", "test_vmlp.npy"), 0.157, False),
    "xcepvid": (os.path.join(W, "work", "uec_k1", "test_xcepvid.npy"), 0.138, False),
    "vcnn":    (os.path.join(W, "work", "uec_k2", "test_vcnn.npy"), 0.009, False),
}
VOTES = ",".join([os.path.join(W, "acco", "download", "sub_d7.csv") + ":1.0",
                  os.path.join(W, "public_subs", "nomannic19__ts-emb-3wdc-temporal-fusion-ensemble", "submission.csv") + ":0.5",
                  os.path.join(W, "acco", "download", "sub_d1.csv") + ":0.4",
                  os.path.join(W, "public_subs", "abhinavm2811__3rd-wear-dataset-challenge-hasca-2026", "submission.csv") + ":0.4"])
E19 = [(os.path.join(W, "exp", "full", "v3b_full", "test.npy"), 0.5), (os.path.join(W, "exp", "full", "v1_full", "test.npy"), 0.3),
       (os.path.join(W, "exp", "aka", "aka_full", "test.npy"), 0.45)]

def load(p):
    P = np.load(p).astype(np.float64); assert P.shape == (12234, 19), (p, P.shape)
    P = np.clip(P, 0, None); return P / np.maximum(P.sum(1, keepdims=True), 1e-12)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--decode", action="store_true"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    ref = pd.read_csv(os.path.join(SUBS, "sub_e30_vote8.csv")).sort_values("id").target_feature.to_numpy()
    from sklearn.metrics import f1_score
    have = {}
    for k, (p, w, imu) in MEMBERS.items():
        if os.path.exists(p) and (a.only is None or k in a.only.split(",")):
            have[k] = (load(p), w, imu); P = have[k][0]
            print(f"{k:8s} w={w:.3f} agree_e30 {np.mean(P.argmax(1)==ref):.3f} F1_vs_e30 {f1_score(ref, P.argmax(1), average='macro'):.3f} null {np.mean(P.argmax(1)==0):.3f}")
        else: print(f"{k:8s} MISSING")
    if not have: sys.exit("no members")
    def ens(sel):
        ws = np.array([have[k][1] for k in sel]); ws = ws / ws.sum()
        return sum(w * have[k][0] for k, w in zip(sel, ws))
    U = ens(list(have)); np.save(os.path.join(WORK, "uec_ens.npy"), U.astype(np.float32))
    print(f"uec_ens ({','.join(have)}) agree_e30 {np.mean(U.argmax(1)==ref):.3f} F1_vs_e30 {f1_score(ref, U.argmax(1), average='macro'):.3f} null {np.mean(U.argmax(1)==0):.3f}")
    imu_sel = [k for k in have if have[k][2]]
    if imu_sel:
        Ui = ens(imu_sel); np.save(os.path.join(WORK, "uec_imu.npy"), Ui.astype(np.float32))
        print(f"uec_imu ({','.join(imu_sel)}) agree_e30 {np.mean(Ui.argmax(1)==ref):.3f} F1_vs_e30 {f1_score(ref, Ui.argmax(1), average='macro'):.3f}")
    if not a.decode: return
    py = sys.executable; env = dict(os.environ, PYTHONPATH=os.path.join(W, "shim"))
    def run(name, extra, votes=VOTES, norm=None, tags="fusion_v1:0.2"):
        probs = os.path.join(WORK, f"probs_{name}.npy")
        cmd = [py, os.path.join(W, "src", "make_probs.py"), probs, "--tags", tags, "--extra", extra]
        if votes: cmd += ["--votes", votes]
        if norm: cmd += ["--norm_total", str(norm)]
        subprocess.run(cmd, check=True, env=env)
        out = os.path.join(SUBS, f"sub_{name}.csv")
        subprocess.run([py, os.path.join(W, "exp", "transductive", "refine_ns.py"), "--probs", probs, "--out", out], check=True, env=env, cwd=os.path.join(W, "exp", "transductive"))
        lab = pd.read_csv(out).sort_values("id").target_feature.to_numpy()
        print(f"==> {name}: diff vs e30 {np.mean(lab != ref):.3f}, null {np.mean(lab==0):.3f}, classes {len(np.unique(lab))}")
    e19x = ",".join(f"{p}:{w}" for p, w in E19); ue = os.path.join(WORK, "uec_ens.npy")
    run("u1_e19_uec05", e19x + f",{ue}:0.5")                      # e19 + uec at 0.5
    run("u2_e19_uec10", e19x + f",{ue}:1.0")                      # e19 + uec at 1.0
    run("u3_uec_primary", f"{ue}:1.0,{E19[0][0]}:0.4,{E19[2][0]}:0.3", tags="none")   # uec primary + v3b + aka
    run("u4_uec_only_votes", f"{ue}:1.45", tags="none")           # uec only (+ votes), sharpness matched to e19 total

if __name__ == "__main__":
    main()
