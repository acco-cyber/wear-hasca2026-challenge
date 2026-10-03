"""Evaluate a grouping configuration from cached score matrices (scores/<model>_<session>.npz).
usage: python eval_cfg.py <model> <session> [...]      env: SMOOTH=k:lam  REFINE=w  ANCHOR_SEED=int
Outputs: pairwise Hungarian table (exact / same-label, dyn / static), 4-way grouping, anchor-centric assignment,
leave-one-session-out calibrated confidence -> precision/coverage table, and calib_<tag>.npz for run_test.py."""
import sys, os, time, json, numpy as np
sys.path.insert(0, r"E:\Claude code\wear\exp\hyb\agents\group")
from common import PAIRS, LIMBS, hungarian
from grouping import (knn_matrix, smooth_scores, anchor_assign, finish_feats, is_sym_pair, fit_logistic,
                      predict_logistic, CONF_FEATS)

HERE = r"E:\Claude code\wear\exp\hyb\agents\group"
CACHE = os.path.join(HERE, "cache"); SC = os.path.join(HERE, "scores")
model = sys.argv[1]; sessions = sys.argv[2:]
_sm = os.environ.get("SMOOTH", "0:0").split(":"); SMOOTH_K, SMOOTH_LAM = int(_sm[0]), float(_sm[1])
REFINE = float(os.environ.get("REFINE", "0.5")); SEED = int(os.environ.get("ANCHOR_SEED", "7"))
tag = model + (f"_s{SMOOTH_K}x{SMOOTH_LAM:g}" if SMOOTH_K > 0 else "") + (f"_r{REFINE:g}" if REFINE != 0.5 else "")


def acc_split(ok, same, mask):
    return dict(exact=float(ok[mask].mean()) if mask.any() else float("nan"),
                same=float(same[mask].mean()) if mask.any() else float("nan"), n=int(mask.sum()))


rng = np.random.default_rng(SEED)
results = {}
R = dict(sess=[], sec=[], limb=[], sym=[], same=[], exact=[], lab=[], dyn_a=[], F=[])
for name in sessions:
    t0 = time.time()
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    lab = z["lab"]; N = len(lab); dyn = [z[f"dyn{l}"] for l in range(4)]; desc = [z[f"desc{l}"] for l in range(4)]
    sz = np.load(os.path.join(SC, f"{model}_{name}.npz"))
    S = {(a, b): sz[f"S{a}{b}"].astype(np.float32) for (a, b) in PAIRS}
    if SMOOTH_K > 0:
        P = [knn_matrix(desc[l], SMOOTH_K) for l in range(4)]
        S = {p: smooth_scores(S[p], P[p[0]], P[p[1]], SMOOTH_LAM) for p in PAIRS}
    _, cnt = np.unique(lab, return_counts=True); chance = float(((cnt / N) ** 2).sum())
    res = {"N": int(N), "chance_same": chance, "pairs": {}}
    truth = np.arange(N)
    for (a, b) in PAIRS:
        asg = hungarian(S[(a, b)])
        ok = asg == truth; same = lab[asg] == lab
        r = {k: acc_split(ok, same, m) for k, m in [("all", np.ones(N, bool)), ("dyn", dyn[a]), ("sta", ~dyn[a])]}
        res["pairs"][f"{LIMBS[a]}-{LIMBS[b]}"] = r
        print(f"{name} {LIMBS[a]:9s}-{LIMBS[b]:9s} exact all {r['all']['exact']:.3f} dyn {r['dyn']['exact']:.3f} sta {r['sta']['exact']:.3f} | "
              f"same-label all {r['all']['same']:.3f} dyn {r['dyn']['same']:.3f} sta {r['sta']['same']:.3f} | chance {chance:.3f} n_dyn {r['dyn']['n']} n_sta {r['sta']['n']}", flush=True)
    leg = hungarian(S[(1, 3)]); arm = hungarian(S[(0, 2)])
    S4 = S[(0, 1)].T + S[(0, 3)][:, leg].T + S[(1, 2)][:, arm] + S[(2, 3)][arm][:, leg].T
    g = hungarian(S4)
    grp = np.stack([g, truth, arm[g], leg], 1)
    glab = np.stack([lab[grp[:, l]] for l in range(4)], 1)
    res["group4"] = {"exact4": float((grp == grp[:, :1]).all(1).mean()), "same4": float((glab == glab[:, :1]).all(1).mean()),
                     "pairwise_same": float((glab[:, :, None] == glab[:, None, :]).mean())}
    print(f"{name} 4-way: exact {res['group4']['exact4']:.3f} all-same-label {res['group4']['same4']:.3f} pairwise-same {res['group4']['pairwise_same']:.3f}", flush=True)
    # anchor-centric (one identified limb per second, uniformly random)
    anc = rng.integers(0, 4, size=N); anc_row = truth.copy()
    is_anchor = [anc == m for m in range(4)]
    groups, feats = anchor_assign(S, anc, anc_row, is_anchor, refine=REFINE)
    feats = finish_feats(S, groups, feats, anc, dyn)
    ex = []; sm = []; dy = []
    for m in range(4):
        sel = np.where(anc != m)[0]; r_ = groups[sel, m]
        e = r_ == sel; s_ = lab[r_] == lab[sel]; d_ = feats[sel, m, 4] > 0.5
        ex.append(e); sm.append(s_); dy.append(d_)
        R["sess"] += [name] * len(sel); R["sec"] += sel.tolist(); R["limb"] += [m] * len(sel)
        R["sym"] += [is_sym_pair(m, anc[s]) for s in sel]; R["same"] += s_.tolist(); R["exact"] += e.tolist()
        R["lab"] += lab[sel].tolist(); R["dyn_a"] += d_.tolist(); R["F"].append(feats[sel, m, :])
    ex = np.concatenate(ex); sm = np.concatenate(sm); dy = np.concatenate(dy)
    r = {k: acc_split(ex, sm, mk) for k, mk in [("all", np.ones(len(ex), bool)), ("dyn", dy), ("sta", ~dy)]}
    res["anchor"] = r
    print(f"{name} anchor-centric: exact {r['all']['exact']:.3f} same-label all {r['all']['same']:.3f} dyn {r['dyn']['same']:.3f} sta {r['sta']['same']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    results[name] = res

# ---------------- confidence calibration (leave-one-session-out) ----------------
sess = np.array(R["sess"]); is_sym = np.array(R["sym"]); same = np.array(R["same"]); labs = np.array(R["lab"])
dyn_a = np.array(R["dyn_a"]); secs = np.array(R["sec"]); F = np.concatenate(R["F"]).astype(np.float64)
mu = F.mean(0); sd = F.std(0) + 1e-6; Fz = (F - mu) / sd
conf = np.zeros(len(F))
for name in sessions:
    te = sess == name
    for kind in (True, False):
        tr = (~te) & (is_sym == kind); ts = te & (is_sym == kind)
        if ts.sum() == 0 or tr.sum() == 0: continue
        w = fit_logistic(Fz[tr], same[tr].astype(float))
        conf[ts] = predict_logistic(w, Fz[ts])
if len(sessions) == 1:  # no LOSO possible: in-sample (optimistic) just for debugging
    for kind in (True, False):
        sel = is_sym == kind
        conf[sel] = predict_logistic(fit_logistic(Fz[sel], same[sel].astype(float)), Fz[sel])
print("\n=== anchor-centric assignments: same-label precision vs coverage (LOSO-calibrated confidence) ===")
print(f"all slots: n={len(conf)} same-label {same.mean():.3f} (sym {same[is_sym].mean():.3f} n={is_sym.sum()}, cross {same[~is_sym].mean():.3f} n={(~is_sym).sum()})")
table = {}
key = np.array([f"{s}:{c}" for s, c in zip(sess, secs)])
ukey, inv = np.unique(key, return_inverse=True)
for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    keep = conf >= thr
    row = dict(thr=thr, coverage=float(keep.mean()), precision=float(same[keep].mean()) if keep.any() else float("nan"),
               cov_sym=float(keep[is_sym].mean()), prec_sym=float(same[keep & is_sym].mean()) if (keep & is_sym).any() else float("nan"),
               cov_cross=float(keep[~is_sym].mean()), prec_cross=float(same[keep & ~is_sym].mean()) if (keep & ~is_sym).any() else float("nan"),
               prec_dyn=float(same[keep & dyn_a].mean()) if (keep & dyn_a).any() else float("nan"),
               prec_sta=float(same[keep & ~dyn_a].mean()) if (keep & ~dyn_a).any() else float("nan"))
    n_assigned = 1 + np.bincount(inv, weights=keep.astype(float), minlength=len(ukey))
    cnts = np.bincount(n_assigned.astype(int), minlength=5)[1:5] / len(ukey)
    row["tiles_with_1234_limbs"] = cnts.tolist()
    table[str(thr)] = row
    print(f"thr {thr:.2f}: coverage {row['coverage']:.3f} precision {row['precision']:.3f} | sym cov {row['cov_sym']:.3f} prec {row['prec_sym']:.3f} | "
          f"cross cov {row['cov_cross']:.3f} prec {row['prec_cross']:.3f} | dyn {row['prec_dyn']:.3f} sta {row['prec_sta']:.3f} | tiles with 1/2/3/4 limbs {np.round(cnts,3)}")
bins = np.linspace(0, 1, 11); b = np.clip(np.digitize(conf, bins) - 1, 0, 9)
print("reliability (conf bin -> observed same-label rate, n):", [(round(float(same[b == k].mean()), 2), int((b == k).sum())) if (b == k).any() else None for k in range(10)])
keep = conf >= 0.8
pa = {l: (float(same[keep & (labs == l)].mean()) if (keep & (labs == l)).any() else float("nan"), float(keep[labs == l].mean())) for l in np.unique(labs)}
print("per-activity at thr 0.8 (precision, coverage):", {k[:16]: (round(v[0], 2), round(v[1], 2)) for k, v in sorted(pa.items())})
W = {}
for kind in (True, False):
    sel = is_sym == kind
    W["sym" if kind else "cross"] = fit_logistic(Fz[sel], same[sel].astype(float))
np.savez(os.path.join(HERE, f"calib_{tag}.npz"), mu=mu, sd=sd, w_sym=W["sym"], w_cross=W["cross"], feats=np.array(CONF_FEATS))
print("logistic weights (intercept + standardised feats", CONF_FEATS, ") sym:", np.round(W["sym"], 2), "cross:", np.round(W["cross"], 2))
results["precision_coverage"] = table
with open(os.path.join(HERE, f"eval_{tag}.json"), "w") as f:
    json.dump(results, f, indent=1)
print("saved", tag)
