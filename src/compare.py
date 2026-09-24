"""Agreement matrix between candidate submissions and known-scored files."""
import sys, numpy as np, pandas as pd
W = r"E:\Claude code\wear"

def load_sub(p):
    return pd.read_csv(p).sort_values("id").iloc[:, 1].to_numpy().astype(int)

def known():
    L = np.load(W + r"\work\lgbm_v1\test.npy"); F = np.load(W + r"\work\fusion_v1\test.npy")
    raw = (0.8 * np.log(np.clip(L, 1e-6, 1)) + 0.2 * np.log(np.clip(F, 1e-6, 1))).argmax(1)
    return {"raw_blend": raw,
            "d7(.697)": load_sub(W + r"\acco\download\sub_d7.csv"), "d3(.691)": load_sub(W + r"\acco\download\sub_d3.csv"),
            "d1(.615)": load_sub(W + r"\acco\download\sub_d1.csv"), "aka(.670)": load_sub(W + r"\public_subs\akhyar2612__0-670\submission.csv"),
            "hong": load_sub(W + r"\public_subs\honghanhhh__wear-hasca\submission.csv"),
            "nom(.613)": load_sub(W + r"\public_subs\nomannic19__ts-emb-3wdc-temporal-fusion-ensemble\submission.csv"),
            "uda": load_sub(W + r"\public_subs\udaken10__base-line-liner-model\submission.csv")}

if __name__ == "__main__":
    S = {}
    for p in sys.argv[1:]:
        S[p.split("\\")[-1].replace("sub_", "")[:22]] = load_sub(p)
    S.update(known()); k = list(S)
    pd.set_option("display.width", 250)
    print(pd.DataFrame([[round((S[a] == S[b]).mean(), 3) for b in k] for a in k], index=k, columns=k).to_string())
    print("null rates", {a: round(float((S[a] == 0).mean()), 3) for a in k})
