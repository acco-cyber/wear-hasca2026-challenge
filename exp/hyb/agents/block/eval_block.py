"""Task 2: block-prediction accuracy of label-free estimators on OOF activity windows (+ caches p1 per method)."""
import os, sys, time
import numpy as np
from common import *
import methods as M
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof_p1.npz")
sm, P = load_oof(); y, fold, sbj, rec = sm["y"], sm["fold"], sm["sbj"], sm["rec"]
emb = np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32)
fo = dict(np.load(os.path.join(KEEP, "feat_oof.npz")))
l0 = np.load(os.path.join(KEEP, "links_L0.npz")); succ = l0["oof_succ"].astype(np.int64); score = l0["oof_score"].astype(np.float32)
a_ = y > 0; tb = BLK[y]; th = true_half(sm)
p_self = M.m_self(P); hard = a_ & ((p_self > 0.5) != (tb == 1))
print(f"activity windows {a_.sum()}, self-block wrong (hard) {hard.sum()}", flush=True)
CFGS = {
    "self": ("m_self", {}),
    "km2_resid": ("m_kmeans", dict(k=2, feat="resid")),
    "km2_cent": ("m_kmeans", dict(k=2, feat="cent")),
    "km4_resid": ("m_kmeans", dict(k=4, feat="resid")),
    "km8_resid": ("m_kmeans", dict(k=8, feat="resid")),
    "km16_resid": ("m_kmeans", dict(k=16, feat="resid")),
    "km8_cent": ("m_kmeans", dict(k=8, feat="cent")),
    "km16_cent": ("m_kmeans", dict(k=16, feat="cent")),
    "spec_resid": ("m_spectral", dict(feat="resid")),
    "spec_raw": ("m_spectral", dict(feat="raw")),
    "prop_link": ("m_prop", dict(graph="link", alpha=0.99, iters=200)),
    "prop_link_a9": ("m_prop", dict(graph="link", alpha=0.9, iters=50)),
    "prop_knn_raw": ("m_prop", dict(graph="knn", alpha=0.99, iters=100, feat="raw")),
    "prop_knn_resid": ("m_prop", dict(graph="knn", alpha=0.99, iters=100, feat="resid")),
    "prop_knn_resid_a9": ("m_prop", dict(graph="knn", alpha=0.9, iters=50, feat="resid")),
    "prop_both_raw": ("m_prop", dict(graph="both", alpha=0.99, iters=200, feat="raw")),
    "prop_both_resid": ("m_prop", dict(graph="both", alpha=0.99, iters=200, feat="resid")),
    "st_video": ("m_selftrain", dict(featset="video")),
    "st_video_lofo": ("m_selftrain", dict(featset="video", lofo=True)),
    "st_all": ("m_selftrain", dict(featset="all")),
    "ns_knn": ("m_nullscene", dict(seed="prop_knn_resid_a9")),
    "ns_knn_c2": ("m_nullscene", dict(seed="prop_knn_resid_a9", conf=0.2)),
    "ns_knn_all": ("m_nullscene", dict(seed="prop_knn_resid_a9", pc_on="all")),
    "ns_link": ("m_nullscene", dict(seed="prop_link_a9")),
    "ns_both": ("m_nullscene", dict(seed="prop_both_raw")),
}
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CFGS)
cache = dict(np.load(OUT)) if os.path.exists(OUT) else {}
for nm in only:
    fn, kw = CFGS[nm]; t0 = time.time(); kw = dict(kw)
    if "seed" in kw:
        kw["seed_p1"] = cache[kw.pop("seed")]
    p1 = getattr(M, fn)(P, emb=emb, feats=fo, sbj=sbj, succ=succ, score=score, **kw)
    cache[nm] = p1; np.savez(OUT, **cache)
    pb = p1 > 0.5
    acc = np.mean(pb[a_] == (tb[a_] == 1)); acc_all = np.mean(pb == (th == 1))
    fix = np.sum(hard & (pb == (tb == 1))); brk = np.sum(a_ & ~hard & (pb != (tb == 1)))
    perm = max(acc, 1 - acc)
    conf = np.abs(p1 - 0.5)[a_].mean()
    per_s = [np.mean(pb[a_ & (sbj == s)] == (tb[a_ & (sbj == s)] == 1)) for s in np.unique(sbj)]
    acc_null = np.mean(pb[~a_] == (th[~a_] == 1))
    print(f"{nm:18s} act-acc {acc:.4f} (perm {perm:.4f}) null-rows-vs-true-half {acc_null:.4f} | hard fixed {fix}/{hard.sum()} "
          f"| broke {brk} | mean|p1-.5| {conf:.3f} | worst sbj acc {np.min(per_s):.3f} [{time.time()-t0:.0f}s]", flush=True)
