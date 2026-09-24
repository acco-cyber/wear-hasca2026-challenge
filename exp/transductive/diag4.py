"""Cross-chain kNN purity: neighbours restricted to windows outside the query's own chain (non-local information)."""
import os, pickle
from tlib import *
from sklearn.neighbors import NearestNeighbors

C = pickle.load(open(os.path.join(TD, "cache_eval.pkl"), "rb")); C.update(pickle.load(open(os.path.join(TD, "cache_extra.pkl"), "rb")))
rows = []
for s, d in C.items():
    y = d["y"]; n = d["n"]; ch = d["chains"]; cid = np.zeros(n, int)
    for i, c in enumerate(ch): cid[c] = i
    lens = np.array([len(c) for c in ch]); r = dict(s=s, nch=len(ch), med_len_w=float(np.median(lens[cid])), acc0=round((d["lab0"] == y).mean(), 3))
    X = build_X(d["F"], "v768", 64, 32); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    _, idx = NearestNeighbors(n_neighbors=101).fit(Xn).kneighbors(Xn); idx = idx[:, 1:]
    other = cid[idx] != cid[:, None]
    for k in (5, 10):
        # first k neighbours from other chains
        sel = np.full((n, k), -1)
        for i in range(n):
            o = idx[i][other[i]][:k]; sel[i, :len(o)] = o
        ok = sel >= 0; r[f"xch_k{k}_pur"] = round(float((y[np.where(ok, sel, 0)] == y[:, None])[ok].mean()), 3)
        # neighbour-vote of decoded labels from other chains vs own decoded label
        lab = d["lab0"]; votes = np.zeros((n, NC))
        for j in range(k): m = ok[:, j]; votes[np.where(m)[0], lab[sel[m, j]]] += 1
        v = votes.argmax(1); r[f"xch_k{k}_vote_acc"] = round(float((v == y).mean()), 3)
        r[f"xch_k{k}_fix"] = round(float(((v == y) & (lab != y)).mean()), 3); r[f"xch_k{k}_break"] = round(float(((v != y) & (lab == y)).mean()), 3)
    rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows); pd.set_option("display.width", 300); print(df.to_string(index=False)); print(df.mean(numeric_only=True).round(3).to_dict())
