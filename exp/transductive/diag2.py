"""kNN neighbour label purity within subject (true labels) for several feature spaces vs. the link graph."""
import sys, os, pickle
from tlib import *
from refine_core import knn_graph
from sklearn.neighbors import NearestNeighbors

def main():
    C = pickle.load(open(os.path.join(TD, "cache_eval.pkl"), "rb")); C.update(pickle.load(open(os.path.join(TD, "cache_extra.pkl"), "rb")))
    rows = []
    for s, d in C.items():
        y = d["y"]; n = d["n"]; r = dict(s=s)
        # link graph purity (weighted)
        pur = [];
        for a, (idx, w) in enumerate(d["g"]):
            if len(idx): pur.append((w * (y[idx] == y[a])).sum() / w.sum())
        r["link_pur"] = round(float(np.mean(pur)), 3)
        for spec in ("v768", "vm+vs+vd", "vm", "vm+vs+vd+imu", "imu"):
            X = build_X(d["F"], spec, 64, 32); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            nn = NearestNeighbors(n_neighbors=31).fit(Xn); _, idx = nn.kneighbors(Xn); idx = idx[:, 1:]
            for k in (5, 20):
                r[f"{spec}_k{k}"] = round(float((y[idx[:, :k]] == y[:, None]).mean()), 3)
        # raw (non-whitened) v768 cosine
        V = d["F"]["v768"]; V = V - V.mean(0); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        _, idx = NearestNeighbors(n_neighbors=21).fit(Vn).kneighbors(Vn); r["raw768_k20"] = round(float((y[idx[:, 1:]] == y[:, None]).mean()), 3)
        rows.append(r); print(r, flush=True)
    df = pd.DataFrame(rows); pd.set_option("display.width", 300); print(df.to_string(index=False)); print(df.mean(numeric_only=True).round(3).to_dict())

if __name__ == "__main__":
    main()
