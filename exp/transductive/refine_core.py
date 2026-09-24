"""Per-subject transductive refinement: pseudo-labels from the decoded result -> subject-own classifier -> log-space blend
with the base probabilities -> re-decode. Shared by the sim harness (exp_refine.py) and the test CLI (refine.py)."""
import numpy as np
from tlib import build_X, decode, segments, NC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

DEFAULT = dict(spec="v768", d_vid=64, d_imu=32, sel="marg", q=0.5, seglen=5, clf="lr", C=0.1, k=20, alpha=0.9,
               w=0.5, eps=0.1, rounds=1, K=4, seed=0, Qsrc="P", perclass=True)

def select(lab, Pg, chains, cfg):
    """Returns boolean mask of pseudo-labelled windows and their labels (= decoded labels)."""
    n = len(lab); lp = np.sort(np.log(np.clip(Pg, 1e-9, 1)), 1); marg = lp[:, -1] - lp[:, -2]
    agree = lab == Pg.argmax(1)
    if cfg["sel"] == "all": return np.ones(n, bool)
    if cfg["sel"] == "agree": return agree
    if cfg["sel"] == "marg":
        m = np.zeros(n, bool)
        if cfg["perclass"]:            # top-q by margin within each decoded class (keeps rare classes represented)
            for c in np.unique(lab):
                ii = np.where(agree & (lab == c))[0]
                if len(ii) == 0: continue
                thr = np.quantile(marg[ii], 1 - cfg["q"]); m[ii[marg[ii] >= thr]] = True
        else:
            thr = np.quantile(marg[agree], 1 - cfg["q"]); m = agree & (marg >= thr)
        return m
    if cfg["sel"] == "seg":
        seglen = np.zeros(n, int)
        for l, idx in segments(lab, chains): seglen[idx] = len(idx)
        return agree & (seglen >= cfg["seglen"])
    raise ValueError(cfg["sel"])

def chain_folds(chains, n, K, seed):
    rng = np.random.RandomState(seed); f = np.zeros(n, int)
    for c in chains: f[c] = rng.randint(K)
    return f

def _fit_predict(clf, X, y, Xp, cfg):
    cls = np.unique(y); Q = np.zeros((len(Xp), NC))
    if len(cls) < 2: Q[:, cls] = 1; return Q
    if clf == "lr": m = LogisticRegression(C=cfg["C"], max_iter=300).fit(X, y)
    elif clf == "lda": m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    else: raise ValueError(clf)
    Q[:, m.classes_] = m.predict_proba(Xp); return Q

def knn_graph(X, k):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Xn); d, idx = nn.kneighbors(Xn)
    idx = idx[:, 1:]; sim = np.clip(1 - d[:, 1:] ** 2 / 2, 0, None)            # cosine similarity
    from scipy.sparse import csr_matrix
    n = len(X); W = csr_matrix((sim.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T); dg = np.asarray(W.sum(1)).ravel() + 1e-9
    from scipy.sparse import diags
    Dm = diags(1 / np.sqrt(dg)); return Dm @ W @ Dm, W

def label_spread(S, Y0, alpha, iters=30):
    F = Y0.copy()
    for _ in range(iters): F = alpha * (S @ F) + (1 - alpha) * Y0
    return F

def subject_Q(X, lab, Pg, P, chains, cfg, Xcache=None):
    n = len(lab); m = select(lab, Pg, chains, cfg); y = lab
    if cfg["clf"] in ("lr", "lda"):
        if cfg["K"] and cfg["K"] > 1:
            f = chain_folds(chains, n, cfg["K"], cfg["seed"]); Q = np.zeros((n, NC))
            for kf in range(cfg["K"]):
                tr = m & (f != kf); te = f == kf
                Q[te] = _fit_predict(cfg["clf"], X[tr], y[tr], X[te], cfg)
        else:
            Q = _fit_predict(cfg["clf"], X[m], y[m], X, cfg)
    elif cfg["clf"] == "knn":          # label spreading of one-hot pseudo labels over within-subject kNN graph
        S, _ = knn_graph(X, cfg["k"]); Y0 = np.zeros((n, NC)); Y0[np.where(m)[0], y[m]] = 1
        F = label_spread(S, Y0, cfg["alpha"]); sm = F.sum(1, keepdims=True)
        Q = np.where(sm > 1e-9, F / np.maximum(sm, 1e-12), 1.0 / NC)
    elif cfg["clf"] == "ksmooth":      # spread soft base probabilities (P or Pg) over the kNN graph
        S, _ = knn_graph(X, cfg["k"]); Y0 = (P if cfg["Qsrc"] == "P" else Pg).copy()
        F = label_spread(S, Y0, cfg["alpha"]); Q = F / F.sum(1, keepdims=True)
    elif cfg["clf"] == "ncm":          # nearest class mean (whitened space) -> softmax of -dist^2/2
        cls = np.unique(y[m]); mu = np.stack([X[m & (y == c)].mean(0) for c in cls])
        d2 = ((X[:, None, :] - mu[None]) ** 2).sum(2); L = -0.5 * d2 / cfg.get("tau", 1.0)
        L -= L.max(1, keepdims=True); E = np.exp(L); Q = np.zeros((n, NC)); Q[:, cls] = E / E.sum(1, keepdims=True)
    else: raise ValueError(cfg["clf"])
    return Q, m

def knn_label_Q(X, lab, k=5, alpha=0.9, eps=0.1):
    """Spread one-hot decoded labels of ALL windows over the within-subject kNN graph; returns eps-smoothed Q (n,NC)."""
    n = len(lab); S, _ = knn_graph(X, k); Y0 = np.zeros((n, NC)); Y0[np.arange(n), lab] = 1
    F = label_spread(S, Y0, alpha); Q = F / np.maximum(F.sum(1, keepdims=True), 1e-12)
    return (1 - eps) * Q + eps / NC

def augment_graph(g, X, k, beta):
    """Union of the link graph with a within-subject video kNN graph (cosine, symmetrised); kNN edge weight = beta*cos."""
    _, W = knn_graph(X, k); W = W.tocsr(); out = []
    for a, (idx, w) in enumerate(g):
        lo, hi = W.indptr[a], W.indptr[a + 1]; ki = W.indices[lo:hi]; kw = beta * W.data[lo:hi]
        out.append((np.concatenate([idx, ki]).astype(np.int64), np.concatenate([w, kw])))
    return out

def refine(P, F, g, chains, lab0, Pg0, cfg, y=None, log=None):
    """Returns list of labels per round (round 0 = baseline) and the final blended P.
    cfg["steps"] = [step_cfg, ...] -> sequential steps whose log-Q terms ACCUMULATE: L = logP + sum_i w_i log Q_i; step i's
    pseudo-labels / Pg come from decoding after step i-1."""
    if cfg.get("steps"):
        logP = np.log(np.clip(P, 1e-6, 1)); L = logP.copy(); labs = [lab0]; lab, Pg = lab0, Pg0; Pn = P
        for st in cfg["steps"]:
            sc = {**DEFAULT, **st}; Fx = {**F, "lp": logP} if "lp" in sc["spec"] else F
            X = build_X(Fx, sc["spec"], sc["d_vid"], sc["d_imu"])
            Q, m = subject_Q(X, lab, Pg, P, chains, sc)
            if log is not None and y is not None:
                log.append(dict(n_sel=int(m.sum()), sel_acc=float((lab == y)[m].mean()) if m.any() else np.nan, q_acc=float((Q.argmax(1) == y).mean())))
            Q = (1 - sc["eps"]) * Q + sc["eps"] / NC; L = L + sc["w"] * np.log(Q)
            Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
            lab, Pg, _ = decode(Pn, g, chains); labs.append(lab)
        return labs, Pn
    cfg = {**DEFAULT, **cfg}
    if "lp" in cfg["spec"].split("+"): F = {**F, "lp": np.log(np.clip(P, 1e-6, 1))}
    if cfg["spec"] == "lp": X = build_X(F, "lp")
    else: X = build_X(F, cfg["spec"], cfg["d_vid"], cfg["d_imu"])
    labs = [lab0]; lab, Pg = lab0, Pg0; logP = np.log(np.clip(P, 1e-6, 1)); Pn = P
    if cfg.get("aug_k"):                 # decode on the augmented graph first (round "a"), later rounds use it too
        g = augment_graph(g, X, cfg["aug_k"], cfg["aug_beta"])
        lab, Pg, _ = decode(P, g, chains); labs.append(lab)
        if log is not None and y is not None: log.append(dict(round=0, n_sel=0, sel_acc=np.nan, q_acc=np.nan))
        if cfg["clf"] == "none": return labs, P
    for r in range(cfg["rounds"]):
        Q, m = subject_Q(X, lab, Pg, P, chains, cfg)
        if log is not None and y is not None:
            log.append(dict(round=r + 1, n_sel=int(m.sum()), sel_acc=float((lab == y)[m].mean()) if m.any() else np.nan,
                            q_acc=float((Q.argmax(1) == y).mean())))
        Q = (1 - cfg["eps"]) * Q + cfg["eps"] / NC
        L = logP + cfg["w"] * np.log(Q); Pn = np.exp(L - L.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
        lab, Pg, _ = decode(Pn, g, chains); labs.append(lab)
    return labs, Pn
