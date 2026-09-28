"""Graph stage of the Hanbat pipeline with OUR decoder's labels integrated three ways, CV-able on their OOF rows:
  --prior m    : in the self-training rounds, Q <- (1-m)*Q + m*onehot(ours) (rows with our labels)
  --counts c   : Sinkhorn class-count targets per subject <- (1-c)*default(97/exercise) + c*counts(ours)
  --gate tau[:rule]: after finish, where calibrated confidence < tau [and rule top2|agree], take our label
python graph_lab.py cv  <oof_logp_b.npy> <labels.pkl> [options]      (their OOF rows, L0 links proxy)
python graph_lab.py test <test_logp_b.npy> <ours.csv> <out.csv> [options]   (L2 qn links)"""
import os, sys, argparse, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hanbat_stack as H
from hanbat_stack import (KEEP, HB, N_CLS, CFG, TRAIN_SETS, TEST_CFGS, macro_f1, _norm_rows, label_prop_generic, link_W,
                          smooth_emb, SubjectKNN, parse_cfg)
W = r"E:\Claude code\wear"; HYB = os.path.join(W, "exp", "hyb")


def default_targets(sbj, sets):
    T = {}
    for s in np.unique(sbj):
        n = int((sbj == s).sum()); null = max(n - 18 * CFG["per_ex"] * sets.get(int(s), 1), CFG["null_min"] * n)
        t = np.full(N_CLS, (n - null) / 18.0); t[0] = null; T[int(s)] = t
    return T


def calibrate_targets(P, sbj, targets, iters=50):
    out = np.empty_like(P)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); t_ = targets[int(s)]; Q = P[ii].copy()
        for _ in range(iters):
            Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]; Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out


class Runner2(H.Runner):
    def __init__(self, dd, targets, OH, m, K=40, b=0.0):
        super().__init__(dd, K=K, b=b); self.targets = targets; self.OH = OH; self.m = m

    def cal(self, P, T):
        return calibrate_targets(_norm_rows(P ** T), self.d["sbj"], self.targets)

    def __call__(self, cfg):
        c = dict(alpha=0.95, beta=0.5, tau=0.1, k=10, iters=20, use_p=0.5, link_mode="row", emb_g=0.0, rounds=1, T=2.0,
                 w=0.5, alpha2=None); c.update(cfg)
        knn = self.knn(c["emb_g"])
        kw = dict(use_p=c["use_p"], k=c["k"], tau=c["tau"], iters=c["iters"], norm="row", Lw=self.Lw, beta=c["beta"],
                  link_mode=c["link_mode"], key="p0")
        k1 = (c["emb_g"], c["alpha"], repr(tuple(sorted(kw.items(), key=lambda x: x[0]))))
        if k1 not in self.stage1:
            self.stage1[k1] = label_prop_generic(self.P0, knn, alpha=c["alpha"], **kw)
        F = self.stage1[k1]; a2 = c["alpha2"] if c["alpha2"] is not None else c["alpha"]
        for _ in range(c["rounds"] - 1):
            Q = self.cal(F, c["T"])
            if self.OH is not None and self.m > 0:
                Q = _norm_rows((1 - self.m) * Q + self.m * self.OH)
            F = label_prop_generic(_norm_rows(self.P0 ** (1 - c["w"]) * Q ** c["w"]), knn, Qsrc=self.P0, alpha=a2, **kw)
        return F


def graph_P2(dd, cfg_names, targets, OH, m):
    Ps, runners = [], {}
    for c in sorted((parse_cfg(nm) for nm in cfg_names), key=lambda c: c["link_b"]):
        b = c.pop("link_b")
        if b not in runners:
            runners.clear(); runners[b] = Runner2(dd, targets, OH, m, b=b)
        Ps.append(runners[b](c).astype(np.float32))
    return _norm_rows(np.mean(Ps, 0))


def finish2(P, sbj, targets):
    Q = np.clip(P, 1e-12, None) ** (1 / CFG["sharpen_T"]); Q = calibrate_targets(Q / Q.sum(1, keepdims=True), sbj, targets)
    return Q.argmax(1), Q


def make_targets(sbj, sets, ours, c, full_cover):
    T = default_targets(sbj, sets)
    if c <= 0:
        return T
    for s in np.unique(sbj):
        m = sbj == s
        if not full_cover[int(s)]:
            continue
        cnt = np.bincount(ours[m], minlength=N_CLS).astype(np.float64)
        T[int(s)] = (1 - c) * T[int(s)] + c * cnt
    return T


def apply_gate(base, Q, ours, has, tau, rule, other=None, sbj=None):
    conf = Q.max(1)
    if isinstance(tau, str) and tau.startswith("q"):        # per-subject quantile: gate the lowest q fraction
        q = float(tau[1:]); g = np.zeros(len(conf), bool)
        for s in np.unique(sbj):
            m = sbj == s; thr = np.quantile(conf[m], q); g[m] = conf[m] < thr
        g &= has
    else:
        g = has & (conf < float(tau))
    for r in rule.split("+"):
        if r == "top2":
            srt = np.argsort(-Q, 1); g &= (srt[:, 0] == ours) | (srt[:, 1] == ours)
        elif r == "agree" and other is not None:
            g &= (other == ours)
    lab = base.copy(); lab[g] = ours[g]; return lab, g


def run(dd, ours, has, a, eval_y=None, tag="", other=None):
    sets = dd["sets"]; sbj = dd["sbj"]
    cover = {int(s): bool(has[sbj == s].all()) for s in np.unique(sbj)}
    targets = make_targets(sbj, sets, np.where(has, ours, 0), a.counts, cover)
    OH = None
    if a.prior > 0:
        OH = np.full((len(sbj), N_CLS), 1.0 / N_CLS); eps = 0.1
        OH[has] = eps / N_CLS; OH[np.flatnonzero(has), ours[has]] += 1 - eps
    P = graph_P2(dd, TEST_CFGS, targets, OH, a.prior)
    base, Q = finish2(P, sbj, targets)
    lab = base
    if a.gate:
        tau, rule = (a.gate.split(":") + ["plain"])[:2]
        lab, g = apply_gate(base, Q, ours, has, tau, rule, other=other, sbj=sbj)
        if eval_y is None:
            print(f"gated {g.mean():.4f}, per subject " + str({int(s): round(float(g[sbj == s].mean()), 3) for s in np.unique(sbj)}))
    if eval_y is not None:
        S = has
        print(f"{tag} prior={a.prior} counts={a.counts} gate={a.gate}: F1(all) {macro_f1(eval_y, lab):.4f} F1(S) {macro_f1(eval_y[S], lab[S]):.4f}"
              f"  [no-gate F1(S) {macro_f1(eval_y[S], base[S]):.4f}]", flush=True)
    return lab, P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["cv", "test"]); ap.add_argument("logp"); ap.add_argument("ours")
    ap.add_argument("out", nargs="?"); ap.add_argument("--prior", type=float, default=0.0); ap.add_argument("--counts", type=float, default=0.0)
    ap.add_argument("--gate", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--other", default="", help="labels for the 'agree' rule: .npy (cv, their rows) or .csv (test)")
    a = ap.parse_args()
    other = None
    if a.other:
        other = np.load(a.other) if a.other.endswith(".npy") else pd.read_csv(a.other).sort_values("id").target_feature.to_numpy().astype(int)
    if a.mode == "cv":
        sm = {k: v.astype(np.int64) for k, v in np.load(os.path.join(KEEP, "sim_meta.npz")).items()}
        l0 = np.load(os.path.join(KEEP, "links_L0.npz"))
        dd = dict(logp=np.load(a.logp).astype(np.float32), emb=np.load(os.path.join(KEEP, "oof_emb.npy")).astype(np.float32),
                  grp=sm["sbj"], sbj=sm["sbj"], succ=l0["oof_succ"].astype(np.int64), score=l0["oof_score"].astype(np.float32), sets=TRAIN_SETS)
        rows = np.load(os.path.join(HYB, "rows.npz")); o2t = rows["ours_to_theirs"]
        R = pickle.load(open(a.ours, "rb")); ours = np.full(len(sm["y"]), -1, np.int64)
        for s, d in R.items():
            ours[o2t[int(d["a"]):int(d["a"]) + int(d["n"])]] = np.asarray(d["lab"])
        has = ours >= 0
        run(dd, np.where(has, ours, 0), has, a, eval_y=sm["y"], tag=a.tag or os.path.basename(a.logp), other=other)
    else:
        bl = np.load(os.path.join(KEEP, "blend.npz")); l2 = np.load(os.path.join(KEEP, "links_L2_test.npz"))
        sbj = bl["test_sbj"].astype(np.int64)
        dd = dict(logp=np.load(a.logp).astype(np.float32), emb=np.load(os.path.join(KEEP, "test_emb.npy")).astype(np.float32),
                  grp=sbj, sbj=sbj, succ=l2["succ"].astype(np.int64), score=l2["score_qn"].astype(np.float32), sets={})
        ours = pd.read_csv(a.ours).sort_values("id").target_feature.to_numpy().astype(int); has = np.ones(len(ours), bool)
        lab, P = run(dd, ours, has, a, other=other)
        ids = np.load(os.path.join(KEEP, "test_id.npy"), allow_pickle=True); assert (ids == np.arange(len(ids))).all()
        pd.DataFrame({"id": ids, "target_feature": lab.astype(int)}).to_csv(a.out, index=False)
        np.save(a.out.replace(".csv", "_P.npy"), P)
        ref = bl["labels"]; print(f"wrote {a.out}; agreement with kernel labels {np.mean(lab == ref):.4f}; with ours {np.mean(lab == ours):.4f}; null {np.mean(lab == 0):.3f}")


if __name__ == "__main__":
    main()
