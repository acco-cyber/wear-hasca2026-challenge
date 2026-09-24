"""Same-label rate of graph neighbours: link-graph (baseline) vs kNN in video descriptors."""
from lk import *
from feat2 import nrm
S = pickle.load(open(os.path.join(WORK, "sim_struct.pkl"), "rb")); meta, imu, vid = load_prep()
rows = []
for s in EVAL:
    st = S[s]; n = st["n"]; y = st["y"]; V = np.asarray(vid[st["a"]:st["b"]], np.float32); Vn = nrm(V)
    g = build_graph(st["cand"], st["lo"], n, k=10)
    agree = np.concatenate([(y[idx] == y[i]) for i, (idx, w) in enumerate(g)]); wagree = np.concatenate([(y[idx] == y[i]) * w / w.sum() for i, (idx, w) in enumerate(g) if len(idx)])
    r = dict(session=s, link_graph=agree.mean(), link_graph_w=wagree.sum() / n)
    mp = nrm(Vn.mean(1)); Vc = V.mean(1) - V.mean(1).mean(0); mpc = nrm(Vc)
    for nm, D in (("mp", mp), ("mp_centered", mpc)):
        Sm = D @ D.T; np.fill_diagonal(Sm, -9)
        for k in (5, 10, 20):
            nb = np.argpartition(-Sm, k, 1)[:, :k]; r[f"{nm}_k{k}"] = (y[nb] == y[:, None]).mean()
    rows.append(r)
df = pd.DataFrame(rows); print(df.round(3).to_string()); print(df.mean(numeric_only=True).round(3).to_dict())
