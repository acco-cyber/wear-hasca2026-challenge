"""Subject-adaptive (transductive, label-free) features: for every tile, how its video embedding / its blend logits relate
to the INTERIOR tiles of its own subject (tiles > 4 chain steps from any predicted label change, i.e. confidently
labelled by the pipeline): kNN fraction predicted null / predicted current label / predicted best activity, and
cosine to the subject's null and class prototypes. Saves proto.npz (Fo, Ft, names)."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from ne_common import HERE, load_all
sys.path.insert(0, r"E:\Claude code\wear\exp\v4")
import v4_local as V
from v4_local import N_CLS

T0 = time.time()
KS = (10, 30)


def unit(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)


def feats_space(E, lab, qa, interior, sb, nm):
    """E: per-tile vectors (unit rows). kNN among the subject's interior tiles (self excluded)."""
    n = len(lab); out = {}
    for k in KS:
        for t in ("null", "lab", "qa"):
            out[f"{nm}_k{k}_{t}"] = np.zeros(n)
    out[f"{nm}_cos_null"] = np.zeros(n); out[f"{nm}_cos_lab"] = np.zeros(n); out[f"{nm}_cos_qa"] = np.zeros(n)
    for s in np.unique(sb):
        ii = np.flatnonzero(sb == s); ref = ii[interior[ii]]
        Er = E[ref]; lr = lab[ref]
        S = E[ii] @ Er.T                                         # (n_s, n_ref)
        self_ = interior[ii]
        if self_.any():                                           # exclude the tile itself
            pos = np.searchsorted(ref, ii[self_]); S[np.flatnonzero(self_), pos] = -np.inf
        kmax = max(KS); nb = np.argpartition(-S, kmax, axis=1)[:, :kmax]
        sv = np.take_along_axis(S, nb, 1); order = np.argsort(-sv, 1); nb = np.take_along_axis(nb, order, 1)
        L = lr[nb]
        for k in KS:
            out[f"{nm}_k{k}_null"][ii] = (L[:, :k] == 0).mean(1)
            out[f"{nm}_k{k}_lab"][ii] = (L[:, :k] == lab[ii][:, None]).mean(1)
            out[f"{nm}_k{k}_qa"][ii] = (L[:, :k] == qa[ii][:, None]).mean(1)
        protos = np.zeros((N_CLS, E.shape[1]))
        for c in range(N_CLS):
            m = lr == c
            if m.any():
                protos[c] = unit(Er[m].mean(0, keepdims=True))[0]
        C = E[ii] @ protos.T
        out[f"{nm}_cos_null"][ii] = C[:, 0]; out[f"{nm}_cos_lab"][ii] = C[np.arange(len(ii)), lab[ii]]
        out[f"{nm}_cos_qa"][ii] = C[np.arange(len(ii)), qa[ii]]
    out[f"{nm}_cos_d"] = out[f"{nm}_cos_null"] - out[f"{nm}_cos_qa"]
    return out


def build(d, z, split):
    o = split == "oof"
    lab = (d["ref_o"] if o else d["ref_t"]).astype(np.int64); qa = (z["qao"] if o else z["qat"]).astype(np.int64)
    sb = (d["oof_sbj"] if o else d["test_sbj"]).astype(np.int64); interior = (z["cdo"] if o else z["cdt"]) > 4
    emb = np.mean([np.load(os.path.join(k, ("oof_emb.npy" if o else "test_emb.npy"))).astype(np.float32) for k in (V.W + r"\work\v4\wear-v4-big-pool-opt-s7\keep4",)], 0)
    Ew = unit(V.whitened_subject(emb, sb, 0.5, 0.1))
    f = feats_space(Ew, lab, qa, interior, sb, "vid")
    B = (d["Bo"] if o else d["Bt"]).astype(np.float64); lw = (d["lwo"] if o else d["lwt"]).astype(np.float64)
    for M, nm in ((B, "blg"), (lw, "wlg")):
        X = M - M.mean(1, keepdims=True)                     # centred log-probs, per-subject standardised
        Xs = np.zeros_like(X)
        for s in np.unique(sb):
            ii = sb == s; Xs[ii] = (X[ii] - X[ii].mean(0)) / (X[ii].std(0) + 1e-6)
        f.update(feats_space(unit(Xs), lab, qa, interior, sb, nm))
    names = list(f.keys())
    print(f"[{time.time() - T0:.0f}s] {split}: {len(names)} proto features, interior {interior.mean():.3f}", flush=True)
    return np.stack([f[k] for k in names], 1).astype(np.float32), names


def main():
    d = load_all(); z = np.load(os.path.join(HERE, "feats.npz"))
    Fo, names = build(d, z, "oof"); Ft, _ = build(d, z, "test")
    np.savez_compressed(os.path.join(HERE, "proto.npz"), Fo=Fo, Ft=Ft, names=np.array(names))
    y = d["oof_y"]; lab = d["ref_o"]; idx = z["cdo"] <= 4
    from sklearn.metrics import roc_auc_score
    for j, nm in enumerate(names):
        a = idx & (lab > 0); b = idx & (lab == 0)
        print(f"{nm:16s} AUC null | act-labelled cand {roc_auc_score(y[a] == 0, Fo[a, j]):.3f}  null-labelled cand {roc_auc_score(y[b] == 0, Fo[b, j]):.3f}")


if __name__ == "__main__":
    main()
