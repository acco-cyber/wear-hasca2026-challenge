"""Diagnostics: (1) pseudo-label quality of the baseline decode; (2) within-subject separability of feature sets with TRUE
labels (chain-grouped 4-fold CV, LR) - the ceiling of any per-subject refinement in that feature space."""
import sys, os, time, pickle
from tlib import *
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

def chain_folds(chains, n, K=4, seed=0):
    rng = np.random.RandomState(seed); f = np.zeros(n, int)
    for c in chains: f[c] = rng.randint(K)
    return f

def main():
    C = pickle.load(open(os.path.join(TD, "cache_eval.pkl"), "rb")); C.update(pickle.load(open(os.path.join(TD, "cache_extra.pkl"), "rb")))
    rows = []
    for s, d in C.items():
        y = d["y"]; lab = d["lab0"]; Pg = d["Pg0"]; n = d["n"]
        lp = np.sort(np.log(Pg), 1); marg = lp[:, -1] - lp[:, -2]; agree = lab == Pg.argmax(1)
        segs = segments(lab, d["chains"]); seglen = np.zeros(n, int)
        for l, idx in segs: seglen[idx] = len(idx)
        r = dict(s=s, base=round(d["f0"], 4), acc=round((lab == y).mean(), 3), agree=round(agree.mean(), 3), acc_agree=round((lab == y)[agree].mean(), 3))
        for q in (0.5, 0.7):
            m = agree & (marg > np.quantile(marg[agree], 1 - q)); r[f"acc_m{q}"] = round((lab == y)[m].mean(), 3)
        for L in (5, 15):
            m = seglen >= L; r[f"cov_seg{L}"] = round(m.mean(), 3); r[f"acc_seg{L}"] = round((lab == y)[m].mean(), 3)
        folds = chain_folds(d["chains"], n)
        for spec in ("vm+vs+vd", "imu", "vm+vs+vd+imu", "v768"):
            X = build_X(d["F"], spec, d_vid=64, d_imu=32); pr = np.zeros(n, int)
            for k in range(4):
                tr = folds != k; m = LogisticRegression(C=0.1, max_iter=300).fit(X[tr], y[tr]); pr[~tr] = m.predict(X[~tr])
            r[f"orc_{spec}"] = round(f1_score(y, pr, average="macro"), 3)
        rows.append(r); print(r, flush=True)
    df = pd.DataFrame(rows); pd.set_option("display.width", 250); print(df.to_string(index=False)); print(df.mean(numeric_only=True).round(4).to_dict())

if __name__ == "__main__":
    main()
