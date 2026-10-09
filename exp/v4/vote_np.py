"""NumPy-only majority votes over finished decodes (no LightGBM / pandas / sklearn), scored out-of-fold; picks K diverse
top votes (each new pick must differ from the already picked ones on >= MIN_DIFF test tiles) and writes their CSVs.
  python vote_np.py [K]"""
import os, sys, csv, itertools
import numpy as np
W = r"E:\Claude code\wear"; S = os.path.join(W, "subs"); N_CLS = 19
st = np.load(os.path.join(W, "work", "v4", "wear-v4-big-pool-opt-s7", "keep4", "stage.npz"), allow_pickle=True)
y, fold = st["oof_y"].astype(int), st["oof_fold"].astype(int)
TAGS = ["v4c_w25_f4n_clog", "v4c_w25_f6_clog", "v4c_w25_f4n_clog_x2", "v4c_w25_f2n_clog", "v4c_w25_f4nu", "v4c_w25_f6ab_clog",
        "v4c_w25_f2n_w46", "v4c_w25_f4n", "v4c_w25_f2n", "v4c_w25_f3q", "v4c_w25_f2u", "v4c_w25_f4n_clog_x4", "v4l_w25_k9n", "v4l_w25_k7n"]
SUB = {t: (np.load(os.path.join(S, f"sub_{t}_labo.npy")).astype(int), np.load(os.path.join(S, f"sub_{t}_labt.npy")).astype(int)) for t in TAGS}
SUBMITTED = {"v4c_w25_f4n_clog", "v4c_w25_f6_clog", "v4c_w25_f4n_clog_x2", "v4c_w25_f2n_clog", "v4c_w25_f4nu", "v4c_w25_f2n_w46",
             "v4c_w25_f4n", "v4c_w25_f2n", "v4c_w25_f3q", "v4c_w25_f2u", "v4l_w25_k9n"}


def macro_f1(t, p):
    cm = np.bincount(t * N_CLS + p, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float((2 * tp[ok] / den[ok]).mean())


def vote(names, sp):
    L = np.stack([SUB[n][sp] for n in names]); n = L.shape[1]
    c = np.zeros((n, N_CLS)); np.add.at(c, (np.tile(np.arange(n), len(names)), L.ravel()), 1.0)
    c[np.arange(n), L[0]] += 0.5
    return c.argmax(1)


def write(lab, path):
    rows = list(csv.reader(open(os.path.join(W, "data", "sample_submission.csv"))))
    ids = st["ids"].astype(int); assert (ids == np.arange(len(ids))).all() and len(rows) - 1 == len(ids)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(rows[0][:2])
        for r in rows[1:]:
            w.writerow([r[0], int(lab[int(r[0])])])


K = int(sys.argv[1]) if len(sys.argv) > 1 else 7; MIN_DIFF = 15
for t in TAGS:
    print(f"{t:24s} OOF {macro_f1(y, SUB[t][0]):.4f}")
cands = []
lead = ["v4c_w25_f4n_clog", "v4c_w25_f6_clog", "v4c_w25_f2n_clog"]
pool = TAGS[:12]
for r in (3, 5, 7):
    for names in itertools.combinations(pool, r):
        if names[0] not in lead:
            continue
        lab = vote(names, 0); cands.append((macro_f1(y, lab), names))
cands.sort(key=lambda x: -x[0])
print("top OOF votes:", [(round(f, 4), "+".join(n)) for f, n in cands[:5]])
picked, picked_t = [], []
best_t = SUB["v4c_w25_f4n_clog"][1]
prev = [np.array([int(r[1]) for r in list(csv.reader(open(os.path.join(S, p))))[1:]]) for p in ("sub_vote3_best.csv", "sub_vote4_best.csv")]
for f, names in cands:
    lt = vote(names, 1)
    if any((lt != p).sum() < MIN_DIFF for p in picked_t + prev + [best_t]):
        continue
    lo = vote(names, 0); pf = [macro_f1(y[fold == k], lo[fold == k]) for k in range(5)]
    path = os.path.join(S, f"sub_vnp{len(picked) + 1}.csv"); write(lt, path)
    picked.append((round(f, 4), names, [round(x, 4) for x in pf], int((lt != best_t).sum()), path)); picked_t.append(lt)
    if len(picked) == K:
        break
for p in picked:
    print("PICK", p[0], "+".join(p[1]), "per fold", p[2], "| differs from best on", p[3], "->", os.path.basename(p[4]))
