"""Weighted majority vote over submission CSVs. python vote_subs.py out.csv file1:w1 file2:w2 ...
Ties are broken by the first file."""
import sys, numpy as np, pandas as pd
out = sys.argv[1]; specs = [s.rsplit(":", 1) for s in sys.argv[2:]]
labs = []; ws = []
for p, w in specs:
    d = pd.read_csv(p).sort_values("id"); labs.append(d.iloc[:, 1].to_numpy().astype(int)); ws.append(float(w)); ids = d.id.to_numpy()
C = np.zeros((len(ids), 19))
for i, (l, w) in enumerate(zip(labs, ws)):
    C[np.arange(len(l)), l] += w + (1e-3 if i == 0 else 0.0)
v = C.argmax(1)
pd.DataFrame({"id": ids, "target_feature": v}).to_csv(out, index=False)
print("wrote", out, "| agreement with first file", round(float((v == labs[0]).mean()), 4), "| null", round(float((v == 0).mean()), 3))
