"""Re-shape probabilities of a model (prior division / temperature) to match what the decoder expects.
python recal.py <src_dir> <out_prefix> alpha1,alpha2 [T]  -> exp/base/<out_prefix>_a<alpha>/ oof+test, runs sim."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
src = sys.argv[1]; src = src if os.path.isabs(src) else os.path.join(EXP, src)
O = np.load(os.path.join(src, "oof.npy")); T = np.load(os.path.join(src, "test.npy"))
ref = np.load(os.path.join(EXP, "blend_v1f", "oof.npy"))
def stats(P, name):
    Q = P.reshape(-1, NC); Q = Q[~np.isnan(Q[:, 0])]
    ent = -(Q * np.log(np.clip(Q, 1e-9, 1))).sum(1).mean()
    print(f"{name}: mean P(null) {Q[:, 0].mean():.3f} argmax-null {np.mean(Q.argmax(1)==0):.3f} entropy {ent:.3f} maxp {Q.max(1).mean():.3f}", flush=True)
stats(ref, "ref blend"); stats(O, "src")
y = meta().y.to_numpy(); pur = meta().pur.to_numpy()
prior = np.bincount(y[pur >= 0.8], minlength=NC) / (pur >= 0.8).sum()
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
for al in [float(x) for x in sys.argv[3].split(",")]:
    L = (to_log(O) - al * np.log(prior)) / temp; P = norm_probs(L).astype(np.float32)
    Lt = (to_log(T) - al * np.log(prior)) / temp; Pt = norm_probs(Lt).astype(np.float32)
    name = f"{sys.argv[2]}_a{al}_T{temp}"; od = os.path.join(EXP, name); os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "oof.npy"), P); np.save(os.path.join(od, "test.npy"), Pt)
    stats(P, name); f = eval_single(P, name=name)
    s, _ = run_sim(os.path.join(od, "oof.npy"), name); print("SIM", name, "raw", s["raw"], "chain", s["g_chain_cal_ps0.8"], "oracle", s["oracle_cal60_160"], flush=True)
    json.dump(dict(src=src, alpha=al, T=temp, oof_f1=f, sim=s), open(os.path.join(od, "res.json"), "w"), indent=1)
