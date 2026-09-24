import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
W = r"E:\Claude code\wear"
nb = pd.read_csv(W + r"\public_subs\nomannic19__ts-emb-3wdc-temporal-fusion-ensemble\submission.csv").sort_values('id').target_feature.values
best = pd.read_csv(W + r"\subs\sub_transductive_mrf4_e19_aka045_abh.csv").sort_values('id').target_feature.values
print(f"{'file':60s} agree_nb  agree_best  F1_vs_best  F1(nb_vs_best)={f1_score(best, nb, average='macro'):.4f} agree(nb,best)={np.mean(nb==best):.4f}")
for f in sys.argv[1:]:
    P = np.load(f); a = P.argmax(1)
    print(f"{f[-60:]:60s} {np.mean(a==nb):.4f}    {np.mean(a==best):.4f}      {f1_score(best, a, average='macro'):.4f}")
