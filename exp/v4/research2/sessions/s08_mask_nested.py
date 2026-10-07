"""nested OOF evaluation of a soft session-block mask on top of a decode:
   Q'_ic = Q_ic * P(session block == blk(c))^lam for activity classes; a tile's label changes only when argmax Q' differs.
   lam picked per held-out fold on the other 4 folds (nested); pi from s07 (label-free: links diffusion of predicted blocks)."""
from common import *
from s07_session_diffusion import posterior

d = base(); y = d["oof_y"]; fold = d["oof_fold"]
S = os.path.join(W, "subs")
runs = {"K7": (d["QB_OOF"].astype(np.float64), d["ref_oof"].astype(int)),
        "b4wa": (np.load(os.path.join(S, "sub_v4c_b4wa_Qo.npy")).astype(np.float64), np.load(os.path.join(S, "sub_v4c_b4wa_labo.npy")).astype(int))}
pis = np.load(os.path.join(OUT, "s07_pi.npy"), allow_pickle=True)[0]
LAMS = [0, 0.25, 0.5, 1, 2, 4]


def apply(Q, lab, pi, lam):
    pa = np.clip(pi, 1e-3, 1 - 1e-3); w = np.ones_like(Q)
    w[:, A] = pa[:, None] ** lam; w[:, B] = (1 - pa)[:, None] ** lam
    Qn = Q * w; new = lab.copy()
    d0, d1 = Q.argmax(1), Qn.argmax(1); m = d1 != d0          # only tiles whose decision the penalty changes
    new[m] = d1[m]
    return new


for nm, (Q, lab) in runs.items():
    for key in (("links", 0.9), ("links", 0.99), ("links+knn", 0.9)):
        pi = pis[key] if nm == "b4wa" else posterior(lab, *key)
        np.save(os.path.join(OUT, f"s08_pi_{nm}_{key[0]}_{key[1]}.npy"), pi)
        cand = {lam: apply(Q, lab, pi, lam) for lam in LAMS}
        nested = lab.copy(); chosen = []
        for k in range(5):
            tr = fold != k
            best = max(LAMS, key=lambda l: f1(y[tr], cand[l][tr])); chosen.append(best)
            nested[fold == k] = cand[best][fold == k]
        print(f"{nm} pi={key}: base {f1(y, lab):.4f} | fixed lam " + " ".join(f"{l}:{f1(y, cand[l]):.4f}" for l in LAMS)
              + f" | NESTED {f1(y, nested):.4f} ({f1(y, nested) - f1(y, lab):+.4f}) lams {chosen}", flush=True)
