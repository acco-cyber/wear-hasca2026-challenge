"""Fold-honest combined links (L2) for the 69,326 test-like OOF tiles of the train recordings.

Fork of woominyo's public notebook "WEAR@HASCA 2026 | Timeline Reconstruction + Graph" (Apache 2.0), which adapts
honghanhhh's "WEAR@HASCA Hungarian Chain Viterbi LB 0.74" (Apache 2.0). All data/link code below is copied verbatim
from that notebook; only the driver differs:
  * no window models: the OOF window blend log-probs (d["logp"]) and Plp come from our GPU fork's kept output
    (kernel koushikrudra/wear-hanbat-gpu, keep/blend.npz)
  * FIVE link scorers: scorer g is trained on the pair rows of the subjects of the other 4 folds (same rows as the
    notebook: true successor + 16 negatives, features of ridge x{h} for a row of fold h) and applied to all candidate
    pairs of the held-out fold-g subjects (features of ridge x{g}); Hungarian + cycle breaking per subject; scores are
    quantile-normalised per subject against the notebook's fixed QN_REF table (exactly as for test)
Output: /kaggle/working/oof_L2.npz  (succ = global OOF row or -1, score, score_qn, + diagnostics arrays)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import gc, hashlib, json, re, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb

SMOKE = os.environ.get("WEAR_SMOKE", "0") == "1"      # local code test on the subjects whose raw video is local
SMOKE_SUBJECTS = {int(x) for x in os.environ.get("WEAR_SMOKE_SUBJECTS", "5,9,7,8,6").split(",")}
SLUG = "3rd-wear-dataset-challenge-hasca-2026"
ON_KAGGLE = Path("/kaggle/input").exists()
GPU_OK = False                                        # CPU-only kernel: the Gram statistics use numpy


# ---- verbatim: notebook lines 83-92
def find_input():
    if os.environ.get("WEAR_DATA"):
        return Path(os.environ["WEAR_DATA"])
    for p in (Path("/kaggle/input/competitions") / SLUG, Path("/kaggle/input") / SLUG):
        if (p / "test" / "test_meta_data.csv").exists():
            return p
    hits = sorted(Path("/kaggle/input").glob("*/test/test_meta_data.csv")) + \
        sorted(Path("/kaggle/input").glob("*/*/test/test_meta_data.csv"))
    assert hits, "competition data not found: attach the competition or set WEAR_DATA"
    return hits[0].parent.parent

INPUT = find_input()
WORK = Path(os.environ.get("WEAR_WORK", "/kaggle/working" if ON_KAGGLE else "wear_run")).resolve()
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
if os.environ.get("WEAR_ROOT"):
    ROOT = Path(os.environ["WEAR_ROOT"])
elif ON_KAGGLE and shutil.disk_usage("/tmp").free > 12e9:
    ROOT = Path("/tmp/wear")
else:
    ROOT = WORK / "_wear"
PROC = ROOT / "proc"
for _p in (ROOT, PROC):
    _p.mkdir(parents=True, exist_ok=True)
T_START, TIMES = time.time(), {}


# ---- verbatim: notebook lines 114-137
def rss_gb():
    try:
        s = Path("/proc/self/status").read_text()
        return tuple(int(re.search(k + r":\s+(\d+)", s).group(1)) / 2 ** 20 for k in ("VmRSS", "VmHWM"))
    except Exception:
        return float("nan"), float("nan")


class stage:
    """logs wall time and memory of a pipeline stage and frees memory afterwards"""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"\n===== {self.name} | start at {(self.t - T_START) / 60:.1f} min", flush=True)
        return self

    def __exit__(self, *exc):
        gc.collect()
        TIMES[self.name] = time.time() - self.t
        r, h = rss_gb()
        print(f"===== {self.name} | {TIMES[self.name] / 60:.1f} min (total {(time.time() - T_START) / 60:.1f} min) "
              f"| RSS {r:.1f} GB, peak {h:.1f} GB", flush=True)


def find_keep():
    """the kept output folder of the GPU fork (kernel source); WEAR_KEEP overrides"""
    if os.environ.get("WEAR_KEEP"):
        return Path(os.environ["WEAR_KEEP"])
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "blend.npz" in files and "sim_meta.npz" in files:
            hits.append(Path(root))
        dirs[:] = [x for x in dirs if x not in ("train", "test", "inertial_feat", "videomae_feat")]
    print("kernel-source candidates:", hits)
    assert hits, "keep/blend.npz of koushikrudra/wear-hanbat-gpu not found under /kaggle/input"
    return hits[0]


KEEP = find_keep()
print(f"input {INPUT}\nwork {WORK}\nscratch {ROOT}\nkeep {KEEP}\nSMOKE={SMOKE} {sorted(SMOKE_SUBJECTS) if SMOKE else ''}",
      flush=True)
os.system("ls -la " + str(KEEP) if ON_KAGGLE else "echo")

# ---- verbatim: notebook lines 171-182
N_CLS, WIN, VLEN = 19, 50, 15
CFG = {"imu_w": 0.3, "lp": (10, 0.1, 0.5, 0.9), "link_alpha": 0.5, "link_iters": 10, "sharpen_T": 0.5, "per_ex": 97,
       "null_min": 0.05}
TRAIN_SETS = {0: 2, 14: 2}   # train subjects 0 and 14 recorded the exercise circuit twice; every test subject once


def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


# ---- verbatim: notebook lines 273-279
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)",
           "jogging (butt-kicks)", "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)",
           "stretching (hamstrings)", "stretching (lumbar rotation)", "push-ups", "push-ups (complex)",
           "sit-ups", "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LABEL_IDX = {c: i for i, c in enumerate(CLASSES)}
SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC_COLS = [f"{s}_acc_{a}" for s in SENSORS for a in "xyz"]


# ---- verbatim: notebook lines 282-307
def prep():
    recs = []
    csvs = sorted((INPUT / "train" / "inertial_feat").glob("*.csv"))     # this order defines the recording index
    if SMOKE:
        csvs = [c for c in csvs if int(c.stem.split("_")[1]) in SMOKE_SUBJECTS]
    for csv in csvs:
        stem = csv.stem
        df = pd.read_csv(csv, usecols=["sbj_id", *ACC_COLS, "label"], dtype={"label": str})
        acc = df[ACC_COLS].to_numpy(np.float32)
        valid = ~np.isnan(acc).reshape(-1, 4, 3).any(2)
        n_nan = int(np.isnan(acc).sum())
        if n_nan:
            acc = pd.DataFrame(acc).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)
        labels = df["label"].fillna("null").astype(str).str.strip()
        unknown = set(labels.unique()) - set(CLASSES)
        assert not unknown, f"{stem}: unknown labels {unknown}"
        lab = labels.map(LABEL_IDX).to_numpy(np.int8)
        vid = np.load(INPUT / "train" / "videomae_feat" / f"{stem}.npy").astype(np.float16)
        np.save(PROC / f"{stem}_acc.npy", acc)
        np.save(PROC / f"{stem}_lab.npy", lab)
        np.save(PROC / f"{stem}_valid.npy", valid)
        np.save(PROC / f"{stem}_vid.npy", vid)
        recs.append({"stem": stem, "sbj": int(df["sbj_id"].iloc[0]), "T": len(acc), "F": len(vid)})
        print(f"{stem}: T={len(acc)} ({len(acc) / 50:.0f}s) F={len(vid)} nan={n_nan}", flush=True)
        del df, acc, vid
    (PROC / "recordings.json").write_text(json.dumps(recs, indent=1))
    return recs


with stage("W1 preprocessing (train only)"):
    if not (ROOT / ".done_prep").exists():
        recs = prep()
        (ROOT / ".done_prep").touch()
    recs = json.loads((PROC / "recordings.json").read_text())
    if not SMOKE:
        assert len(recs) == 24 and sum(r["T"] for r in recs) == 3466400 and sum(r["F"] for r in recs) == 2079840
        assert all(r["F"] * 5 == r["T"] * 3 for r in recs)
    print(f"{len(recs)} recordings, {len({r['sbj'] for r in recs})} subjects")

# ---- verbatim: notebook lines 343-405
STRIDE, FOLDS = 5, 5


def oof_windows():
    recs = json.loads((PROC / "recordings.json").read_text())
    W_rec, W_start, W_y, W_sbj, W_ok = [], [], [], [], []
    for i, r in enumerate(recs):
        lab = np.load(PROC / f"{r['stem']}_lab.npy").astype(np.int64)
        val = np.load(PROC / f"{r['stem']}_valid.npy")
        T, Fr = r["T"], r["F"]
        starts = np.arange(0, T - WIN + 1, STRIDE)
        starts = starts[starts * 3 // 5 + 8 + VLEN <= Fr]
        cs = np.zeros((T + 1, N_CLS), np.int32)
        np.cumsum(np.eye(N_CLS, dtype=np.int32)[lab], axis=0, out=cs[1:])
        cnt = cs[starts + WIN] - cs[starts]
        miss = np.zeros((T + 1, 4), np.int32)
        np.cumsum(~val, axis=0, out=miss[1:])
        W_ok.append((miss[starts + WIN] - miss[starts]) == 0)
        W_rec.append(np.full(len(starts), i))
        W_start.append(starts)
        W_y.append(cnt.argmax(1))
        W_sbj.append(np.full(len(starts), r["sbj"]))
    W_rec, W_start, W_y, W_sbj, W_ok = map(np.concatenate, (W_rec, W_start, W_y, W_sbj, W_ok))
    keep = W_ok.any(1)
    W_rec, W_start, W_y, W_sbj, W_ok = (x[keep] for x in (W_rec, W_start, W_y, W_sbj, W_ok))

    sbjs, sbj_n = np.unique(W_sbj, return_counts=True)
    fold_of, load = {}, np.zeros(FOLDS)
    for s in sbjs[np.argsort(-sbj_n)]:
        f = int(load.argmin())
        fold_of[s] = f
        load[f] += sbj_n[sbjs == s][0]
    W_fold = np.array([fold_of[s] for s in W_sbj])

    out = {k: [] for k in ("rec", "start", "sensor", "y", "sbj", "fold")}
    for fold in range(FOLDS):
        idx = np.flatnonzero(W_fold == fold)
        idx = idx[W_start[idx] % WIN == 0]
        rng = np.random.default_rng(1234 + fold)
        cp = np.cumsum(W_ok[idx] / W_ok[idx].sum(1, keepdims=True), 1)
        sensor = (rng.random(len(idx))[:, None] > cp).sum(1)
        for k, v in (("rec", W_rec[idx]), ("start", W_start[idx]), ("sensor", sensor), ("y", W_y[idx]),
                     ("sbj", W_sbj[idx]), ("fold", np.full(len(idx), fold))):
            out[k].append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    return recs, out


def load_inputs(recs, w):
    """per-window IMU of the chosen limb (N,50,3) float32 and video (N,15,768) float16"""
    n = len(w["rec"])
    acc = np.zeros((n, WIN, 3), np.float32)
    vid = np.zeros((n, VLEN, 768), np.float16)
    for i, r in enumerate(recs):
        ii = np.flatnonzero(w["rec"] == i)
        if not len(ii):
            continue
        a = np.load(PROC / f"{r['stem']}_acc.npy").reshape(-1, 4, 3)
        v = np.load(PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
        st = w["start"][ii]
        acc[ii] = a[st[:, None] + np.arange(WIN), w["sensor"][ii][:, None]]
        vid[ii] = v[(st * 3 // 5)[:, None] + 8 + np.arange(VLEN)]
    return acc, vid


# ---- verbatim: notebook lines 1193-1196
def true_successor(rec, start):
    """row of the tile that starts 1 s later in the same recording, else -1. TRAINING TARGET / diagnostics only"""
    key = {(r, s): n for n, (r, s) in enumerate(zip(rec.tolist(), start.tolist()))}
    return np.array([key.get((r, s + 50), -1) for r, s in zip(rec.tolist(), start.tolist())], np.int64)


with stage("W2 test-like OOF windows"):
    ksim = {k: v.astype(np.int64) for k, v in dict(np.load(KEEP / "sim_meta.npz")).items()}
    _recs, _w = oof_windows()
    _w = {k: v.astype(np.int64) for k, v in _w.items()}
    if not SMOKE:
        sim = _w
        assert len(sim["y"]) == 69326
        assert np.bincount(sim["fold"]).tolist() == [14353, 12995, 14625, 12768, 14585]
        assert hashlib.sha1(sim["sensor"].tobytes()).hexdigest()[:12] == "f81d90085315"
        assert hashlib.sha1(np.stack([sim["rec"], sim["start"]]).tobytes()).hexdigest()[:12] == "e4177f2589d7"
        for k in ksim:
            assert (ksim[k] == sim[k]).all(), f"sim_meta[{k}] differs from the GPU fork"
        OOF_ROWS = np.arange(len(sim["y"]))
    else:
        # the subset's own fold split / limb draw differ from the full run: take the GPU fork's rows of these subjects
        stems_all = [c.stem for c in sorted((INPUT / "train" / "inertial_feat").glob("*.csv"))]
        loc = {r["stem"]: i for i, r in enumerate(recs)}
        OOF_ROWS = np.flatnonzero(np.isin(ksim["sbj"], sorted(SMOKE_SUBJECTS)))
        sim = {k: v[OOF_ROWS].copy() for k, v in ksim.items()}
        sim["rec"] = np.array([loc[stems_all[r]] for r in sim["rec"]], np.int64)
        a_ = set(zip(sim["rec"].tolist(), sim["start"].tolist()))
        b_ = set(zip(_w["rec"].tolist(), _w["start"].tolist()))
        assert a_ == b_, (len(a_), len(b_), len(a_ ^ b_))
        assert (np.sort(sim["y"]) == np.sort(_w["y"])).all()
    _acc, _vid = load_inputs(recs, sim)
    np.save(ROOT / "sim_vid.npy", _vid)
    np.save(ROOT / "sim_acc.npy", _acc)
    _kacc = np.load(KEEP / "sim_acc.npy", mmap_mode="r")[OOF_ROWS]
    print("sim_acc vs GPU fork: max abs diff", float(np.abs(_kacc - _acc).max()))
    assert np.allclose(_kacc, _acc, atol=1e-5)
    del _acc, _vid, _kacc, _w, ksim
    fold_of = {int(s): int(f) for s, f in zip(sim["sbj"], sim["fold"])}
    print("rows", len(sim["y"]), "folds", {f: sorted(s for s in fold_of if fold_of[s] == f) for f in range(FOLDS)})

with stage("load OOF window blend + Plp"):
    _bl = np.load(KEEP / "blend.npz")
    d = {"logp": _bl["oof_logp"][OOF_ROWS].astype(np.float32), "y": sim["y"], "sbj": sim["sbj"], "fold": sim["fold"],
         "sensor": sim["sensor"],
         "vid": lambda: np.load(ROOT / "sim_vid.npy", mmap_mode="r"),      # (N,15,768) float16, raw VideoMAE
         "acc": lambda: np.load(ROOT / "sim_acc.npy")}                     # (N,50,3) float32, g
    PLP_OOF = _bl["plp_oof"][OOF_ROWS].astype(np.float32)
    del _bl
    assert d["logp"].shape == (len(sim["y"]), N_CLS) and np.isfinite(d["logp"]).all()
    print(f"window blend OOF macro F1 {macro_f1(d['y'], d['logp'].argmax(1)):.4f} (GPU fork full: 0.7253-ish), "
          f"Plp {macro_f1(d['y'], PLP_OOF.argmax(1)):.4f}")
    TRUE_SUCC = true_successor(sim["rec"], sim["start"])
    N_OOF = len(d["y"])

# ---- verbatim: notebook lines 1515-1695
STRIDE_B, D_V = 10, 768
BF, BH, BT, BL, BM = range(5)
SHRINK, IMU_LAM, RIDGE_LAM = 0.1, 1e-3, 0.1


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def bs_vid_blocks(fr):
    """fr (n,15,768) unit + centred frames -> (n, 5*768) [First, Head(3), Tail(3), Last, Mean]"""
    return np.concatenate([fr[:, 0], fr[:, :3].mean(1), fr[:, -3:].mean(1), fr[:, -1], fr.mean(1)], 1)


def bs_imu_summ(a):
    """a (n,50,3) -> tail_in (n,19), head_out (n,11), head_in (n,19), tail_out (n,11)"""
    mag = np.linalg.norm(a, axis=2)
    lsd = lambda m: np.log(m.std(1) + 0.01)[:, None]
    tail_in = np.concatenate([a.mean(1), a.std(1), a[:, -10:].mean(1), a[:, -10:].std(1), a[:, -1], a[:, -3:].mean(1), lsd(mag)], 1)
    head_in = np.concatenate([a.mean(1), a.std(1), a[:, :10].mean(1), a[:, :10].std(1), a[:, 0], a[:, :3].mean(1), lsd(mag)], 1)
    head_out = np.concatenate([a[:, :10].mean(1), a[:, :3].mean(1), a.mean(1), lsd(mag[:, :10]), lsd(mag)], 1)
    tail_out = np.concatenate([a[:, -10:].mean(1), a[:, -3:].mean(1), a.mean(1), lsd(mag[:, -10:]), lsd(mag)], 1)
    return tail_in, head_out, head_in, tail_out


def add1(x):
    return np.concatenate([x, np.ones((len(x), 1), x.dtype)], 1)


class Gram:
    """S += zi^T zi, C += zi^T zj in float64 (fp32 products; on the GPU when available, which saves about 5 min)"""
    def __init__(self, dim):
        if GPU_OK:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            self.torch, self.dev = torch, torch.device("cuda")
            self.S = torch.zeros((dim, dim), dtype=torch.float64, device=self.dev)
            self.C = torch.zeros((dim, dim), dtype=torch.float64, device=self.dev)
        else:
            self.torch, self.S, self.C = None, np.zeros((dim, dim)), np.zeros((dim, dim))

    def add(self, zi, zj):
        if self.torch is None:
            self.S += zi.T @ zi
            self.C += zi.T @ zj
        else:
            a, b = self.torch.from_numpy(zi).to(self.dev), self.torch.from_numpy(zj).to(self.dev)
            self.S += (a.T @ a).double()
            self.C += (a.T @ b).double()

    def result(self):
        if self.torch is None:
            return self.S, self.C
        S, C = self.S.cpu().numpy(), self.C.cpu().numpy()
        del self.S, self.C
        self.torch.cuda.empty_cache()
        return S, C


def subject_stats(recs_):
    """label-free sufficient statistics of consecutive 1-s tile pairs (s, s+50) of one subject's recordings"""
    acc_sum, cnt = np.zeros(D_V), 0
    for r in recs_:
        v = np.load(PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
        st = np.arange(0, r["T"] - WIN + 1, WIN)
        st = st[st * 3 // 5 + 8 + VLEN <= r["F"]]
        idx = ((st * 3 // 5)[:, None] + 8 + np.arange(VLEN)).ravel()
        fr = unit(np.asarray(v[idx], np.float32))
        acc_sum += fr.sum(0)
        cnt += len(fr)
    mu = (acc_sum / cnt).astype(np.float32)
    Dz = 5 * D_V + 1
    gram = Gram(Dz)
    n = 0
    IM = {k: np.zeros(s) for k, s in (("fS", (4, 4, 20, 20)), ("fC", (4, 4, 20, 11)), ("fY", (4, 4, 11, 11)), ("fn", (4, 4)),
                                      ("bS", (4, 4, 20, 20)), ("bC", (4, 4, 20, 11)), ("bY", (4, 4, 11, 11)))}
    for r in recs_:
        v = np.load(PROC / f"{r['stem']}_vid.npy", mmap_mode="r")
        a = np.load(PROC / f"{r['stem']}_acc.npy").reshape(-1, 4, 3)
        val = np.load(PROC / f"{r['stem']}_valid.npy")
        T, Fr = r["T"], r["F"]
        st = np.arange(0, T - 2 * WIN + 1, STRIDE_B)
        st = st[(st + WIN) * 3 // 5 + 8 + VLEN <= Fr]
        frames = unit(np.asarray(v[: Fr], np.float32)) - mu  # (F,768)
        for c0 in range(0, len(st), 4000):
            s = st[c0:c0 + 4000]
            fi = (s * 3 // 5)[:, None] + 8 + np.arange(VLEN)
            zi = add1(bs_vid_blocks(frames[fi]))
            zj = add1(bs_vid_blocks(frames[fi + 30]))  # s + 50 samples = +30 frames
            gram.add(zi, zj)
            n += len(s)
            del zi, zj
        del frames
        miss = np.zeros((T + 1, 4), np.int32)
        np.cumsum(~val, axis=0, out=miss[1:])
        ok_i = (miss[st + WIN] - miss[st]) == 0
        ok_j = (miss[st + 2 * WIN] - miss[st + WIN]) == 0
        su_i = [bs_imu_summ(a[st[:, None] + np.arange(WIN), l]) for l in range(4)]
        su_j = [bs_imu_summ(a[st[:, None] + WIN + np.arange(WIN), l]) for l in range(4)]
        for l in range(4):
            for l2 in range(4):
                m = ok_i[:, l] & ok_j[:, l2]
                if not m.any():
                    continue
                x, y = add1(su_i[l][0][m]), su_j[l2][1][m]
                IM["fS"][l, l2] += x.T @ x
                IM["fC"][l, l2] += x.T @ y
                IM["fY"][l, l2] += y.T @ y
                IM["fn"][l, l2] += m.sum()
                x, y = add1(su_j[l2][2][m]), su_i[l][3][m]
                IM["bS"][l, l2] += x.T @ x
                IM["bC"][l, l2] += x.T @ y
                IM["bY"][l, l2] += y.T @ y
    S, C = gram.result()
    return dict(S=S.astype(np.float32), C=C.astype(np.float32), n=n, **IM)


def blk(*bs):
    return np.concatenate([np.arange(b * D_V, (b + 1) * D_V) for b in bs] + [np.array([5 * D_V])])


XF, YF = blk(BL, BT, BH, BM), np.arange(BH * D_V, (BH + 1) * D_V)  # forward: inputs of window i, target = head of j
XB, YB = blk(BF, BH, BT, BM), np.arange(BT * D_V, (BT + 1) * D_V)  # backward: inputs of window j, target = tail of i


def _dir_stats(st, fwd):
    S, C = st["S"], st["C"]
    if fwd:
        xx, xy, yy = S[np.ix_(XF, XF)], C[np.ix_(XF, YF)], S[np.ix_(YF, YF)]
    else:  # sum z_j z_j^T ~ S, sum z_j z_i^T = C^T
        xx, xy, yy = S[np.ix_(XB, XB)], C[np.ix_(YB, XB)].T, S[np.ix_(YB, YB)]
    return xx, xy, yy, float(st["n"])


def _solve(xx, xy, n, lam):
    reg = lam * np.mean(np.diag(xx)[:-1])
    A_ = xx + reg * np.diag(np.r_[np.ones(len(xx) - 1), 0.0])
    return np.linalg.solve(A_, xy)


def _prec_sqrt(R):
    e, U = np.linalg.eigh((R + R.T) / 2)
    e = np.maximum(e, 0)
    e = e + SHRINK * e.mean()
    return ((U / np.sqrt(e)) @ U.T).astype(np.float32), float(np.sum(np.log(e)))


def fit_ridge(fold_sum, train_folds):
    st = None
    for g in train_folds:
        f = fold_sum[g]
        if st is None:
            st = {k: np.array(v, np.float64) for k, v in f.items()}
        else:
            for k in st:
                st[k] += f[k]
    lam = {True: RIDGE_LAM, False: RIDGE_LAM}       # the value CV picked for every fold model
    m = {"lam_f": lam[True], "lam_b": lam[False]}
    for fwd, tag in ((True, "f"), (False, "b")):
        xx, xy, yy, n = _dir_stats(st, fwd)
        W = _solve(xx, xy, n, lam[fwd])
        R = (yy - xy.T @ W - W.T @ xy + W.T @ xx @ W) / n
        m["W" + tag] = W.astype(np.float32)
        m["P" + tag], _ = _prec_sqrt(R)
        m["r2" + tag] = 1 - np.trace(R) / (np.trace(yy) / n)
    for tag in ("f", "b"):
        Wl = np.zeros((4, 4, 20, 11), np.float32)
        Pl = np.zeros((4, 4, 11, 11), np.float32)
        Ld = np.zeros((4, 4), np.float32)
        for l in range(4):
            for l2 in range(4):
                xx, xy, yy, n = st[tag + "S"][l, l2], st[tag + "C"][l, l2], st[tag + "Y"][l, l2], st["fn"][l, l2]
                reg = IMU_LAM * np.mean(np.diag(xx)[:-1])
                W = np.linalg.solve(xx + reg * np.diag(np.r_[np.ones(19), 0.0]), xy)
                R = (yy - xy.T @ W - W.T @ xy + W.T @ xx @ W) / n
                R = (R + R.T) / 2 + 1e-4 * np.eye(11)
                Pr = np.linalg.inv(R)
                Wl[l, l2], Pl[l, l2], Ld[l, l2] = W, Pr, np.linalg.slogdet(Pr)[1]
        m["imuW" + tag], m["imuP" + tag], m["imuL" + tag] = Wl, Pl, Ld
    return m


with stage("combined links A: ridge boundary predictors"):
    fold_sum = {}
    for g in range(FOLDS):
        acc_ = None
        for sb in sorted(s for s in fold_of if fold_of[s] == g):
            st_ = subject_stats([r for r in recs if r["sbj"] == sb])
            st_ = {k: np.array(v, np.float64) for k, v in st_.items()}
            if acc_ is None:
                acc_ = st_
            else:
                for k in acc_:
                    acc_[k] += st_[k]
            del st_
        if acc_ is None:                                        # SMOKE only: a fold without subjects
            continue
        fold_sum[g] = {k: (v.astype(np.float32) if k in ("S", "C") else v) for k, v in acc_.items()}
        del acc_
        print(f"fold {g}: statistics of {int(fold_sum[g]['n'])} tile pairs [{time.time() - T_START:.0f}s]", flush=True)
    RIDGES = {g: fit_ridge(fold_sum, [h for h in fold_sum if h != g]) for g in fold_sum}
    del fold_sum
    print("ridge r2 (forward, backward):", {g: (round(float(m["r2f"]), 3), round(float(m["r2b"]), 3)) for g, m in RIDGES.items()},
          "  (ours: about 0.58)")

# ---- verbatim: notebook lines 1719-2090
SOURCES = [("th", "r", 40), ("lf", "r", 15), ("mm", "r", 15), ("lfc", "r", 15),
           ("imu", "r", 8), ("imu", "c", 5), ("extlin", "r", 25), ("extlin", "c", 10),
           ("lfz", "c", 30), ("ext2", "r", 25), ("dual", "r", 25)]   # (matrix, row/column top-k, k)
KCAP, KNEW, LOWV = 160, 20, -1e4
OLD30 = ["fwd", "frame", "pooled", "rev", "dom_r", "dom_c", "margin", "rank_r", "rank_c", "same", "gap", "ext",
         "lfc", "thc", "mmc", "revc", "lfc_dom_r", "lfc_dom_c", "ext2", "dratio",
         "bc_w", "bc_lp", "maxp_i", "maxp_j", "magdiff", "stdratio", "limb_i", "limb_j", "gap3", "slope"]
NEW_DECODE = ["lfz", "lfz_dom_r", "lfz_dom_c", "lfz_rank_r", "lfz_rank_c", "extlin", "extlin_rank_r", "extlin_rank_c",
              "ext2_rank_r", "dual", "dual_rank_r", "imu_d", "imu_dom_r", "imu_dom_c", "imu_rank_r", "imu_rank_c",
              "lfc_rank_r", "lfc_rank_c", "n_src", "bc_ca", "maxca_i", "maxca_j", "l1_ca"]
CA_FEATS = ["bc_ca", "maxca_i", "maxca_j", "l1_ca"]           # not used in the 'all' variant
BASE49 = [f for f in OLD30 + NEW_DECODE if f not in CA_FEATS]
NEW16 = ["ll_f", "ll_b", "ll_f_domc", "ll_f_domr", "ll_b_domr", "ll_b_domc", "rs", "rs_domr", "rs_domc", "rs_rank_r",
         "rs_rank_c", "cos_f", "cos_b", "raw_f", "imu_f", "imu_b"]
LK_NEG = 16
LK_ROUNDS = 20 if SMOKE else 300
LK_PARAMS = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                 bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                 num_threads=4, verbosity=-1, seed=0)


def ranks_at(S, cand, axis):
    """rank of S at the candidate positions within its row (axis=1) or column (axis=0): number of larger values"""
    n, K = cand.shape
    ch = np.maximum(cand, 0)
    rows = np.arange(n)[:, None]
    M = S if axis == 1 else S.T
    Ss = np.sort(M, axis=1).astype(np.float64)
    lo, hi = float(Ss.min()), float(Ss.max())
    span = hi - lo + 1.0
    flat = (Ss - lo + (np.arange(n) * span)[:, None]).ravel()
    del Ss
    if axis == 1:
        v, base = S[rows, ch], rows
    else:
        v, base = S[rows, ch], ch
    q = (v.astype(np.float64) - lo + base * span)
    pos = np.searchsorted(flat, q.ravel(), side="right").reshape(n, K)
    return (base + 1) * n - pos


def topk_rows(S, k):
    k = min(k, S.shape[1] - 1)
    return np.argpartition(-S, k - 1, axis=1)[:, :k]


def topk(S, k):
    return np.argpartition(-S, k - 1, axis=1)[:, :k]


def vid_blocks(frc):
    return np.concatenate([frc[:, 0], frc[:, :3].mean(1), frc[:, -3:].mean(1), frc[:, -1], frc.mean(1),
                           np.ones((len(frc), 1), np.float32)], 1)


def imu_summ(a):
    mag = np.linalg.norm(a, axis=2)
    lsd = lambda x: np.log(x.std(1) + 0.01)[:, None]
    tail_in = np.concatenate([a.mean(1), a.std(1), a[:, -10:].mean(1), a[:, -10:].std(1), a[:, -1], a[:, -3:].mean(1), lsd(mag), np.ones((len(a), 1))], 1)
    head_in = np.concatenate([a.mean(1), a.std(1), a[:, :10].mean(1), a[:, :10].std(1), a[:, 0], a[:, :3].mean(1), lsd(mag), np.ones((len(a), 1))], 1)
    head_out = np.concatenate([a[:, :10].mean(1), a[:, :3].mean(1), a.mean(1), lsd(mag[:, :10]), lsd(mag)], 1)
    tail_out = np.concatenate([a[:, -10:].mean(1), a[:, -3:].mean(1), a.mean(1), lsd(mag[:, -10:]), lsd(mag)], 1)
    return [x.astype(np.float32) for x in (tail_in, head_out, head_in, tail_out)]


def imu_pair_ll(summ, limbs, I, J, m):
    """forward / backward IMU log-densities of the candidate pairs (I -> J)"""
    tail_in, head_out, head_in, tail_out = summ
    lf = np.zeros(len(I), np.float32)
    lb = np.zeros(len(I), np.float32)
    li, lj = limbs[I], limbs[J]
    for l in range(4):
        for l2 in range(4):
            k = np.flatnonzero((li == l) & (lj == l2))
            if not len(k):
                continue
            i, j = I[k], J[k]
            r = head_out[j] - tail_in[i] @ m["imuWf"][l, l2]
            lf[k] = -0.5 * np.einsum("nd,de,ne->n", r, m["imuPf"][l, l2], r) + 0.5 * m["imuLf"][l, l2]
            r = tail_out[i] - head_in[j] @ m["imuWb"][l, l2]
            lb[k] = -0.5 * np.einsum("nd,de,ne->n", r, m["imuPb"][l, l2], r) + 0.5 * m["imuLb"][l, l2]
    return lf, lb


def ll_mats(z, m):
    p_ = z[:, XF] @ m["Wf"]
    q = z[:, XB] @ m["Wb"]
    out = []
    for pred, tgt, P, fwd in ((p_, z[:, YF], m["Pf"], True), (q, z[:, YB], m["Pb"], False)):
        a, b = pred @ P, tgt @ P
        L = (b @ a.T) if not fwd else (a @ b.T)  # always [i, j] = i -> j
        L *= 2
        if fwd:
            L -= (a * a).sum(1)[:, None]
            L -= (b * b).sum(1)[None, :]
        else:
            L -= (b * b).sum(1)[:, None]
            L -= (a * a).sum(1)[None, :]
        L *= 0.5 / D_V
        np.fill_diagonal(L, -1e4)
        out.append(L)
    return out, p_, q


def rs_mat(ll_f, ll_b):
    fcm, bcm = ll_f.max(0), ll_b.max(0)
    rs = ll_b + (ll_b - bcm[None]) + (ll_f - fcm[None])
    np.fill_diagonal(rs, -1e5)
    return rs


def new_feats(z, mu, m, cand, summ, limbs):
    """16 ridge-boundary features of the candidate pairs (gathered from (n,n) matrices to save memory)"""
    n, K = cand.shape
    valid = cand >= 0
    ch = np.where(valid, cand, 0)
    rows = np.arange(n)[:, None]
    (ll_f, ll_b), p_, q = ll_mats(z, m)
    F = {"ll_f": ll_f[rows, ch], "ll_b": ll_b[rows, ch]}
    fcm, frm, bcm, brm = ll_f.max(0), ll_f.max(1), ll_b.max(0), ll_b.max(1)
    F["ll_f_domc"] = F["ll_f"] - fcm[ch]
    F["ll_f_domr"] = F["ll_f"] - frm[:, None]
    F["ll_b_domr"] = F["ll_b"] - brm[:, None]
    F["ll_b_domc"] = F["ll_b"] - bcm[ch]
    rs = rs_mat(ll_f, ll_b)
    del ll_f, ll_b
    F["rs"] = rs[rows, ch]
    F["rs_domr"] = F["rs"] - rs.max(1)[:, None]
    F["rs_domc"] = F["rs"] - rs.max(0)[ch]
    rr = np.empty((n, n), np.int32)
    rr[rows, np.argsort(-rs, 1)] = np.arange(n, dtype=np.int32)[None]
    F["rs_rank_r"] = np.log1p(rr[rows, ch])
    rr[np.argsort(-rs, 0), np.arange(n)[None]] = np.arange(n, dtype=np.int32)[:, None]
    F["rs_rank_c"] = np.log1p(rr[rows, ch])
    del rr, rs
    H, T = z[:, YF], z[:, YB]
    S = unit(p_) @ unit(H).T
    F["cos_f"] = S[rows, ch]
    S = unit(T) @ unit(q).T
    F["cos_b"] = S[rows, ch]
    S = unit(p_ + mu) @ unit(H + mu).T
    F["raw_f"] = S[rows, ch]
    del S
    I = np.broadcast_to(rows, cand.shape)[valid]
    J = cand[valid]
    lf, lb = imu_pair_ll(summ, limbs, I, J, m)
    F["imu_f"] = np.zeros((n, K), np.float32)
    F["imu_b"] = np.zeros((n, K), np.float32)
    F["imu_f"][valid], F["imu_b"][valid] = lf, lb
    out = np.stack([F[k] for k in NEW16], 2).astype(np.float32)
    out[~valid] = np.nan
    return out


def subject_all(V, A, limbs, logp, plp, models):
    """one subject: candidate successors (11 sources, cap 160, + ridge top-20 of every model) and pair features.
    Returns cand (n,K) local (-1 pad), valid (n,K), base (P,49) float16, new (len(models),P,16) float32;
    pairs are in row-major order of valid."""
    n = len(V)
    fr = unit(np.asarray(V, np.float32))
    mu = fr.reshape(-1, fr.shape[-1]).mean(0)
    sd = fr.reshape(-1, fr.shape[-1]).std(0) + 1e-3
    frc_all = fr - mu
    z = vid_blocks(frc_all)  # subject-centred unit-frame blocks for the ridge models
    del frc_all
    T, H = unit(fr[:, -3:].mean(1)), unit(fr[:, :3].mean(1))
    last, first, mp = fr[:, -1].copy(), fr[:, 0].copy(), unit(fr.mean(1))
    frc_last, frc_first = fr[:, -1] - mu, fr[:, 0] - mu
    frc_l4, frc_mean = fr[:, -4] - mu, fr.mean(1) - mu
    Tc, Hc = fr[:, -3:].mean(1) - mu, fr[:, :3].mean(1) - mu
    del fr
    a = np.asarray(A, np.float32)
    e = 2 * a[:, -1] - a[:, -2]
    sl = limbs[:, None] == limbs[None, :]

    def mat(name):
        if name == "th":
            S = T @ H.T
        elif name == "lf":
            S = last @ first.T
        elif name == "mm":
            S = mp @ mp.T
        elif name == "lfc":
            S = unit(frc_last) @ unit(frc_first).T
        elif name == "lfz":
            S = unit(frc_last / sd) @ unit(frc_first / sd).T
        elif name == "extlin":
            S = unit(frc_last + (frc_last - frc_l4) * 16 / 3) @ unit(frc_first).T
        elif name == "ext2":
            S = unit(Tc + 1.5 * (Tc - Hc)) @ unit(Hc).T
            S += unit(Tc) @ unit(Hc - 1.5 * (Tc - Hc)).T
        elif name == "thc":
            S = unit(Tc) @ unit(Hc).T
        elif name == "dual":
            L = (unit(frc_last) @ unit(frc_first).T) / 0.05
            np.fill_diagonal(L, LOWV)
            rm = L.max(1, keepdims=True)
            r = rm + np.log(np.exp(L - rm).sum(1, keepdims=True))
            cm = L.max(0, keepdims=True)
            c = cm + np.log(np.exp(L - cm).sum(0, keepdims=True))
            S = 2 * L - r - c
        elif name == "imu":
            S = np.empty((n, n), np.float32)
            for s0 in range(0, n, 1024):
                S[s0:s0 + 1024] = -np.linalg.norm(e[s0:s0 + 1024, None] - a[None, :, 0], axis=2)
            S[~sl] = LOWV
        S = np.asarray(S, np.float32)
        np.fill_diagonal(S, LOWV)
        return S

    # 0) ridge-score top-20 per model
    tops = []
    for m in models:
        (ll_f, ll_b), _, _ = ll_mats(z, m)
        rs = rs_mat(ll_f, ll_b)
        del ll_f, ll_b
        tops.append(np.ascontiguousarray(topk(rs, KNEW)))  # copy: releases the (n,n) int64 argpartition buffer
        del rs
    # 1) similarity-source candidates
    parts = []
    for name, dr, k in SOURCES:
        S = mat(name)
        if dr == "r":
            parts.append(np.ascontiguousarray(topk_rows(S, k)))
        else:
            tk = topk_rows(S.T, k)
            jj = np.repeat(np.arange(n), tk.shape[1])
            parts.append((tk.ravel(), jj))
        del S
    pairs_i = [np.repeat(np.arange(n), p_.shape[1]) if not isinstance(p_, tuple) else p_[0] for p_ in parts]
    pairs_j = [p_.ravel() if not isinstance(p_, tuple) else p_[1] for p_ in parts]
    pi, pj = np.concatenate(pairs_i), np.concatenate(pairs_j)
    ok = pi != pj
    key = pi[ok].astype(np.int64) * n + pj[ok]
    uk, nsrc = np.unique(key, return_counts=True)
    ui, uj = uk // n, uk % n
    cntr = np.bincount(ui, minlength=n)
    Kd = int(min(cntr.max(), KCAP))
    start = np.concatenate([[0], np.cumsum(cntr)])
    # 2) union: source list (capped by number of sources) + ridge-only candidates (n_src = 0)
    lists, nss = [], []
    ext_all = np.concatenate(tops, 1)
    for r in range(n):
        js, ns = uj[start[r]:start[r + 1]], nsrc[start[r]:start[r + 1]]
        if len(js) > Kd:
            o = np.argsort(-ns, kind="stable")[:Kd]
            js, ns = js[o], ns[o]
        ex = np.unique(ext_all[r])
        ex = ex[(ex != r) & ~np.isin(ex, js)]
        lists.append(np.concatenate([js, ex]).astype(np.int32))
        nss.append(np.concatenate([ns, np.zeros(len(ex))]).astype(np.float32))
    K = max(len(x) for x in lists)
    cand = np.full((n, K), -1, np.int32)
    nsr = np.zeros((n, K), np.float32)
    for r in range(n):
        m_ = len(lists[r])
        cand[r, :m_], nsr[r, :m_] = lists[r], nss[r]
    del lists, nss
    valid = cand >= 0
    ch = np.maximum(cand, 0)
    rows = np.arange(n)[:, None]
    # 3) pair features
    F = {"n_src": nsr}
    th = mat("th")
    F["fwd"], F["rev"] = th[rows, ch], th[ch, rows]
    F["dom_r"] = F["fwd"] - th.max(1)[:, None]
    F["dom_c"] = F["fwd"] - th.max(0)[ch]
    srt = np.partition(th, n - 2, axis=1)[:, -2:]
    F["margin"] = np.broadcast_to((srt[:, 1] - srt[:, 0])[:, None], ch.shape)
    F["rank_r"] = np.log1p(ranks_at(th, cand, 1))
    F["rank_c"] = np.log1p(ranks_at(th, cand, 0))
    del th
    F["frame"] = mat("lf")[rows, ch]
    F["pooled"] = mat("mm")[rows, ch]
    S = mat("lfc")
    F["lfc"] = S[rows, ch]
    F["lfc_dom_r"] = F["lfc"] - S.max(1)[:, None]
    F["lfc_dom_c"] = F["lfc"] - S.max(0)[ch]
    F["lfc_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    F["lfc_rank_c"] = np.log1p(ranks_at(S, cand, 0))
    S = mat("thc")
    F["thc"], F["revc"] = S[rows, ch], S[ch, rows]
    del S
    mpc = unit(frc_mean)
    S = mpc @ mpc.T
    F["mmc"] = S[rows, ch]
    del S
    S = mat("ext2")
    F["ext2"] = S[rows, ch]
    F["ext2_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    din = 1 - np.sum(unit(Tc) * unit(Hc), 1)
    F["dratio"] = np.log((1 - F["thc"] + 1e-4) / (0.5 * (din[:, None] + din[ch]) + 1e-4))
    S = mat("lfz")
    F["lfz"] = S[rows, ch]
    F["lfz_dom_r"] = F["lfz"] - S.max(1)[:, None]
    F["lfz_dom_c"] = F["lfz"] - S.max(0)[ch]
    F["lfz_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    F["lfz_rank_c"] = np.log1p(ranks_at(S, cand, 0))
    S = mat("extlin")
    F["extlin"] = S[rows, ch]
    F["extlin_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    F["extlin_rank_c"] = np.log1p(ranks_at(S, cand, 0))
    S = mat("dual")
    F["dual"] = S[rows, ch]
    F["dual_rank_r"] = np.log1p(ranks_at(S, cand, 1))
    S = mat("imu")
    same = sl[rows, ch]
    v = S[rows, ch]
    F["imu_d"] = np.where(same, np.log1p(-v), np.nan)
    F["imu_dom_r"] = np.where(same, np.log1p(-v) - np.log1p(-S.max(1))[:, None], np.nan)
    F["imu_dom_c"] = np.where(same, np.log1p(-v) - np.log1p(-S.max(0))[ch], np.nan)
    F["imu_rank_r"] = np.where(same, np.log1p(ranks_at(S, cand, 1)), np.nan)
    F["imu_rank_c"] = np.where(same, np.log1p(ranks_at(S, cand, 0)), np.nan)
    del S
    pw = np.sqrt(np.exp(logp)).astype(np.float32)
    F["bc_w"] = np.einsum("nc,nkc->nk", pw, pw[ch])
    pl = np.sqrt(plp).astype(np.float32)
    F["bc_lp"] = np.einsum("nc,nkc->nk", pl, pl[ch])
    F["maxp_i"] = np.broadcast_to(plp.max(1)[:, None], ch.shape)
    F["maxp_j"] = plp.max(1)[ch]
    F["same"] = same.astype(np.float32)
    a0 = a[:, 0]
    gap = np.linalg.norm(a[:, -1][:, None] - a0[ch], axis=2)
    ext = np.linalg.norm(e[:, None] - a0[ch], axis=2)
    F["gap"] = np.where(same, np.log1p(gap), np.nan)
    F["ext"] = np.where(same, np.log1p(ext), np.nan)
    g3 = np.linalg.norm(a[:, -3:].mean(1)[:, None] - a[:, :3].mean(1)[ch], axis=2)
    F["gap3"] = np.where(same, np.log1p(g3), np.nan)
    sl_i, sl_j = a[:, -1] - a[:, -4], a[:, 3] - a[:, 0]
    F["slope"] = np.where(same, np.log1p(np.linalg.norm(sl_i[:, None] - sl_j[ch], axis=2)), np.nan)
    mag = np.linalg.norm(a, axis=2)
    F["magdiff"] = np.abs(mag[:, -10:].mean(1)[:, None] - mag[:, :10].mean(1)[ch])
    ls = np.log(mag.std(1) + 0.01)
    F["stdratio"] = np.abs(ls[:, None] - ls[ch])
    F["limb_i"] = np.broadcast_to(limbs[:, None].astype(np.float32), ch.shape)
    F["limb_j"] = limbs[ch].astype(np.float32)
    # the submitted scorer was trained and applied on float16-stored base features: keep that rounding
    base = np.stack([np.asarray(F[k], np.float32)[valid] for k in BASE49], 1).astype(np.float16)
    del F
    summ = imu_summ(a)
    new = np.stack([new_feats(z, mu, m, cand, summ, limbs)[valid] for m in models])
    return cand, valid, base, new


def lk_assign(cl, lo, floor=-50.0):
    """Hungarian 1:1 -> drop invalid links -> break every cycle at its weakest link"""
    n = len(cl)
    cost = np.full((n, n), floor, np.float32)
    valid = cl >= 0
    rows = np.broadcast_to(np.arange(n)[:, None], cl.shape)
    cost[rows[valid], cl[valid]] = lo[valid]
    np.fill_diagonal(cost, -1e4)
    r, c = linear_sum_assignment(-cost)
    succ = np.full(n, -1, np.int64)
    succ[r] = c
    edge = cost[np.arange(n), succ]
    succ[edge <= floor + 1] = -1
    seen = np.zeros(n, bool)
    for s in range(n):
        if seen[s] or succ[s] < 0:
            continue
        path, pos, node = [], {}, s
        while node >= 0 and not seen[node]:
            seen[node] = True
            pos[node] = len(path)
            path.append(node)
            node = int(succ[node])
        if node >= 0 and node in pos:
            cyc = path[pos[node]:]
            succ[cyc[int(np.argmin([cost[x, succ[x]] for x in cyc]))]] = -1
    score = np.where(succ >= 0, cost[np.arange(n), np.maximum(succ, 0)], -50.0).astype(np.float32)
    return succ, score

# ---- verbatim: notebook lines 2130-2228
QN_REF = np.array([
    -6.44053, -4.24589, -3.88974, -3.66993, -3.50726, -3.38501, -3.2956, -3.23938, -3.164, -3.08718, -3.02778, -2.9586,
    -2.91022, -2.85837, -2.81864, -2.76946, -2.73008, -2.69456, -2.66283, -2.62686, -2.58885, -2.56627, -2.53275, -2.5104,
    -2.48715, -2.46204, -2.43572, -2.40964, -2.38207, -2.36088, -2.33957, -2.32001, -2.30017, -2.28296, -2.26387, -2.24349,
    -2.2301, -2.21079, -2.1938, -2.17706, -2.16025, -2.14563, -2.13114, -2.11619, -2.10318, -2.08508, -2.07014, -2.05329,
    -2.03848, -2.02516, -2.00913, -1.99429, -1.98093, -1.96463, -1.95237, -1.93573, -1.92301, -1.90952, -1.89385, -1.88181,
    -1.87227, -1.85927, -1.84582, -1.8301, -1.81951, -1.80524, -1.78992, -1.77961, -1.76906, -1.75497, -1.74413, -1.73459,
    -1.72344, -1.71004, -1.70193, -1.69138, -1.67934, -1.67145, -1.66081, -1.65201, -1.64085, -1.63023, -1.62012, -1.60891,
    -1.60047, -1.59098, -1.58019, -1.56991, -1.5607, -1.55057, -1.54051, -1.52986, -1.52046, -1.51251, -1.5009, -1.49179,
    -1.48359, -1.47493, -1.46435, -1.45481, -1.44453, -1.43285, -1.42277, -1.41318, -1.40498, -1.39849, -1.39163, -1.38326,
    -1.37311, -1.36289, -1.35217, -1.34187, -1.33335, -1.32571, -1.31736, -1.31079, -1.30357, -1.29586, -1.28827, -1.27904,
    -1.27184, -1.26444, -1.25572, -1.2479, -1.23857, -1.23208, -1.2233, -1.21406, -1.20452, -1.19665, -1.18822, -1.17898,
    -1.17071, -1.16304, -1.15494, -1.14773, -1.1397, -1.13294, -1.1259, -1.11888, -1.11101, -1.10422, -1.0951, -1.08625,
    -1.07749, -1.06927, -1.06228, -1.05529, -1.04736, -1.04162, -1.03469, -1.02856, -1.02128, -1.01478, -1.0079, -1.00095,
    -0.99447, -0.98576, -0.97793, -0.97048, -0.96385, -0.95723, -0.95033, -0.94408, -0.93814, -0.93077, -0.92375, -0.91803,
    -0.91354, -0.90606, -0.89776, -0.89092, -0.88521, -0.87876, -0.87191, -0.86506, -0.85759, -0.85028, -0.84307, -0.83672,
    -0.83045, -0.82334, -0.818, -0.81169, -0.80532, -0.79838, -0.79189, -0.78465, -0.77834, -0.77299, -0.76579, -0.75976,
    -0.7529, -0.74708, -0.74097, -0.73445, -0.72724, -0.72146, -0.7151, -0.70933, -0.70393, -0.69712, -0.69097, -0.6841,
    -0.67677, -0.67058, -0.66311, -0.65643, -0.6491, -0.64281, -0.63606, -0.63075, -0.62586, -0.62012, -0.613, -0.60814,
    -0.60173, -0.59573, -0.5902, -0.58485, -0.57897, -0.5706, -0.56592, -0.55923, -0.55203, -0.54653, -0.54152, -0.53428,
    -0.52813, -0.5231, -0.5161, -0.50945, -0.50301, -0.49784, -0.49201, -0.48611, -0.47949, -0.4741, -0.46767, -0.46239,
    -0.45609, -0.44969, -0.44178, -0.43399, -0.42726, -0.4213, -0.41596, -0.41026, -0.40488, -0.39874, -0.39323, -0.38766,
    -0.38296, -0.37716, -0.37223, -0.36639, -0.36116, -0.35422, -0.34778, -0.34205, -0.33703, -0.33198, -0.32623, -0.32018,
    -0.31323, -0.30698, -0.30149, -0.29655, -0.29103, -0.28458, -0.27964, -0.27379, -0.26773, -0.26276, -0.25757, -0.2514,
    -0.2462, -0.23996, -0.23485, -0.22929, -0.22374, -0.21851, -0.21272, -0.20806, -0.20304, -0.19744, -0.19168, -0.1869,
    -0.18204, -0.17571, -0.16855, -0.16364, -0.15844, -0.15309, -0.147, -0.14094, -0.13631, -0.13132, -0.12692, -0.12172,
    -0.11565, -0.10967, -0.10397, -0.09779, -0.09202, -0.08493, -0.07978, -0.07416, -0.06801, -0.06195, -0.05732, -0.05233,
    -0.04683, -0.04122, -0.03647, -0.0315, -0.0264, -0.02113, -0.01609, -0.00972, -0.0044, 0.00074, 0.00664, 0.01336,
    0.01835, 0.02468, 0.03027, 0.03738, 0.04293, 0.04816, 0.05373, 0.05939, 0.06618, 0.07004, 0.07548, 0.08072,
    0.08636, 0.09095, 0.09676, 0.10162, 0.1064, 0.11139, 0.11774, 0.12352, 0.12974, 0.13535, 0.14149, 0.14647,
    0.15119, 0.15616, 0.16146, 0.16594, 0.17061, 0.17593, 0.18111, 0.18694, 0.1925, 0.19721, 0.20228, 0.20668,
    0.21271, 0.21838, 0.22298, 0.22827, 0.23322, 0.23823, 0.24239, 0.24715, 0.25229, 0.25659, 0.2612, 0.26718,
    0.27076, 0.27571, 0.28041, 0.28641, 0.29194, 0.29802, 0.30453, 0.31002, 0.31575, 0.32088, 0.32647, 0.33243,
    0.33765, 0.34191, 0.3469, 0.35169, 0.35634, 0.36184, 0.36637, 0.37168, 0.37655, 0.38114, 0.38578, 0.39042,
    0.39549, 0.40015, 0.40544, 0.41128, 0.41628, 0.42113, 0.42567, 0.43017, 0.43582, 0.4407, 0.44557, 0.44957,
    0.4544, 0.45893, 0.46436, 0.46871, 0.47411, 0.47903, 0.48497, 0.48949, 0.49384, 0.49912, 0.50346, 0.50961,
    0.51492, 0.51972, 0.52429, 0.52964, 0.53516, 0.53935, 0.54457, 0.54957, 0.55513, 0.55969, 0.56451, 0.56936,
    0.57504, 0.5792, 0.58428, 0.58897, 0.59442, 0.59898, 0.60502, 0.60965, 0.61456, 0.61988, 0.62421, 0.62963,
    0.63453, 0.63918, 0.64475, 0.64966, 0.65459, 0.66066, 0.66481, 0.6688, 0.67413, 0.67838, 0.68432, 0.68851,
    0.69351, 0.69814, 0.70398, 0.70894, 0.71447, 0.72007, 0.72499, 0.73021, 0.73478, 0.73977, 0.74392, 0.74968,
    0.75502, 0.76024, 0.76476, 0.76897, 0.77437, 0.77895, 0.78432, 0.78876, 0.79371, 0.79875, 0.80342, 0.80885,
    0.81313, 0.81736, 0.82248, 0.82796, 0.8319, 0.83615, 0.84158, 0.84668, 0.85266, 0.85737, 0.8622, 0.86718,
    0.8729, 0.87703, 0.88082, 0.8857, 0.89172, 0.89664, 0.90113, 0.90599, 0.91114, 0.91492, 0.92079, 0.92673,
    0.9318, 0.93729, 0.94279, 0.94708, 0.95241, 0.95768, 0.9632, 0.9671, 0.97302, 0.97766, 0.98364, 0.98801,
    0.99347, 0.99826, 1.0033, 1.00871, 1.01337, 1.01793, 1.02295, 1.02782, 1.03312, 1.03782, 1.04197, 1.04603,
    1.05149, 1.05612, 1.06054, 1.06531, 1.07025, 1.07495, 1.08051, 1.08563, 1.0902, 1.09462, 1.09911, 1.10393,
    1.10893, 1.11601, 1.12063, 1.12398, 1.12912, 1.13278, 1.13821, 1.14314, 1.14781, 1.15253, 1.15676, 1.16179,
    1.16641, 1.17164, 1.17716, 1.18208, 1.18714, 1.19235, 1.19676, 1.20117, 1.20627, 1.21102, 1.21584, 1.21999,
    1.2254, 1.22995, 1.23481, 1.24022, 1.24548, 1.25046, 1.25449, 1.2604, 1.26603, 1.27027, 1.27627, 1.28115,
    1.28557, 1.2915, 1.2959, 1.3003, 1.3051, 1.30895, 1.31453, 1.32037, 1.32527, 1.3313, 1.33621, 1.34173,
    1.34691, 1.35157, 1.3559, 1.36098, 1.36683, 1.37185, 1.37833, 1.38303, 1.38814, 1.39211, 1.39729, 1.40319,
    1.40914, 1.4143, 1.41758, 1.42195, 1.42761, 1.43294, 1.43723, 1.44242, 1.44813, 1.45319, 1.45862, 1.46404,
    1.47043, 1.47512, 1.48044, 1.48497, 1.49036, 1.49491, 1.49974, 1.50556, 1.51209, 1.51667, 1.52178, 1.52698,
    1.53244, 1.5372, 1.54206, 1.54833, 1.55224, 1.55752, 1.56244, 1.56754, 1.57237, 1.57701, 1.58337, 1.58938,
    1.59431, 1.59936, 1.60361, 1.60822, 1.61354, 1.61922, 1.62492, 1.62997, 1.6348, 1.64083, 1.64626, 1.6517,
    1.65767, 1.66201, 1.66661, 1.67152, 1.67674, 1.68122, 1.6872, 1.69258, 1.69735, 1.70238, 1.70793, 1.71196,
    1.71682, 1.72176, 1.72702, 1.73311, 1.73831, 1.74397, 1.74958, 1.75495, 1.76084, 1.76674, 1.77191, 1.77733,
    1.78194, 1.78626, 1.79196, 1.79752, 1.80365, 1.80872, 1.8133, 1.81963, 1.825, 1.83038, 1.83541, 1.84033,
    1.84517, 1.85058, 1.85513, 1.86069, 1.86485, 1.87113, 1.87566, 1.88111, 1.88662, 1.89142, 1.89772, 1.90311,
    1.90898, 1.91554, 1.92162, 1.92827, 1.93415, 1.93803, 1.94383, 1.94964, 1.95597, 1.96185, 1.96731, 1.97179,
    1.97854, 1.98536, 1.99042, 1.99526, 2.0015, 2.00741, 2.01318, 2.01891, 2.02463, 2.029, 2.03484, 2.04017,
    2.04456, 2.05113, 2.05828, 2.06362, 2.06955, 2.07546, 2.08182, 2.08668, 2.09306, 2.09885, 2.10375, 2.11053,
    2.11625, 2.1203, 2.12569, 2.13197, 2.13789, 2.14285, 2.14689, 2.15386, 2.15935, 2.16474, 2.17041, 2.17683,
    2.18405, 2.1905, 2.19571, 2.20267, 2.20856, 2.21563, 2.22088, 2.22588, 2.23155, 2.23624, 2.24209, 2.24741,
    2.25324, 2.25843, 2.264, 2.26952, 2.27625, 2.28284, 2.28969, 2.29551, 2.30155, 2.3079, 2.315, 2.32125,
    2.32851, 2.3339, 2.34032, 2.34493, 2.35092, 2.35891, 2.3635, 2.36961, 2.3755, 2.38322, 2.38872, 2.39459,
    2.402, 2.40836, 2.41413, 2.42056, 2.4273, 2.43532, 2.4413, 2.4482, 2.4543, 2.46209, 2.4691, 2.47638,
    2.48182, 2.48761, 2.4948, 2.50215, 2.50781, 2.51296, 2.51994, 2.5265, 2.53502, 2.54137, 2.54972, 2.55634,
    2.56183, 2.56712, 2.57428, 2.58167, 2.58688, 2.59453, 2.60224, 2.61022, 2.61741, 2.62364, 2.63063, 2.6386,
    2.64595, 2.65255, 2.65963, 2.66873, 2.67565, 2.68196, 2.68848, 2.69561, 2.70205, 2.70875, 2.71588, 2.72487,
    2.73297, 2.74107, 2.74728, 2.75396, 2.76083, 2.76736, 2.77561, 2.78252, 2.78881, 2.79499, 2.80303, 2.81022,
    2.81784, 2.82591, 2.83398, 2.84127, 2.84885, 2.85811, 2.86715, 2.87343, 2.88316, 2.89153, 2.89885, 2.9076,
    2.91578, 2.92421, 2.93294, 2.94041, 2.94865, 2.95719, 2.96534, 2.97384, 2.98373, 2.99322, 3.00283, 3.00971,
    3.01925, 3.02625, 3.03576, 3.04482, 3.05318, 3.06142, 3.07082, 3.07973, 3.0886, 3.09817, 3.10649, 3.11667,
    3.1249, 3.13408, 3.14267, 3.14989, 3.15681, 3.16492, 3.17209, 3.17978, 3.18945, 3.19895, 3.20858, 3.21667,
    3.2262, 3.2342, 3.24288, 3.25186, 3.26168, 3.26915, 3.27844, 3.28939, 3.29857, 3.3096, 3.31954, 3.32814,
    3.34079, 3.34993, 3.36025, 3.36925, 3.38054, 3.38917, 3.40007, 3.41311, 3.42156, 3.43118, 3.44071, 3.45179,
    3.46286, 3.47391, 3.48499, 3.49936, 3.50833, 3.51937, 3.53001, 3.54127, 3.5534, 3.56458, 3.57572, 3.58738,
    3.59812, 3.60946, 3.62076, 3.63295, 3.64474, 3.65804, 3.66989, 3.68409, 3.69712, 3.7088, 3.72214, 3.73735,
    3.74994, 3.76291, 3.7775, 3.79119, 3.80433, 3.82209, 3.83732, 3.85324, 3.86721, 3.88342, 3.89538, 3.91178,
    3.93119, 3.94805, 3.96147, 3.97768, 3.9942, 4.01005, 4.02693, 4.04245, 4.05933, 4.07904, 4.09915, 4.11498,
    4.13171, 4.15185, 4.16966, 4.18722, 4.20433, 4.22227, 4.24448, 4.26094, 4.28241, 4.30119, 4.32529, 4.34364,
    4.36799, 4.3911, 4.41264, 4.43423, 4.45529, 4.47328, 4.49729, 4.52234, 4.5474, 4.57277, 4.601, 4.62882,
    4.65306, 4.68101, 4.71118, 4.74283, 4.77851, 4.80965, 4.8494, 4.88179, 4.92801, 4.97369, 5.02269, 5.09272,
    5.15093, 5.23979, 5.34928, 5.51486, 6.21687])   # 1001-point quantile table of our pooled linked OOF combined-link scores ('all', 22 subjects)


def qnorm(succ, score, sbj, ref):
    """per subject: rank of each linked score -> quantile of the reference distribution (succ unchanged)"""
    ref = np.sort(np.asarray(ref, np.float64))
    out = score.astype(np.float32).copy()
    for s in np.unique(sbj):
        m = np.flatnonzero((sbj == s) & (succ >= 0))
        if not len(m):
            continue
        r = np.argsort(np.argsort(score[m], kind="stable"), kind="stable")
        u = (r + 0.5) / len(m)
        out[m] = np.quantile(ref, u).astype(np.float32)
    return out


with stage("combined links B: pair features of every train subject (ridge of its own held-out fold)"):
    _V, _A = d["vid"](), d["acc"]()
    SUB, hits = {}, []
    for s in np.unique(d["sbj"]):
        ii = np.flatnonzero(d["sbj"] == s)
        n = len(ii)
        g = fold_of[int(s)]
        cand, valid, base, new = subject_all(_V[ii], _A[ii], d["sensor"][ii], d["logp"][ii], PLP_OOF[ii], [RIDGES[g]])
        tl_ = np.full(N_OOF, -1)
        tl_[ii] = np.arange(n)
        true_loc = np.where(TRUE_SUCC[ii] >= 0, tl_[np.maximum(TRUE_SUCC[ii], 0)], -1)
        rp = np.repeat(np.arange(n), valid.sum(1))
        cj = cand[valid]
        pos = cj == true_loc[rp]
        rng = np.random.default_rng(7000 + int(s))
        key = rng.random(len(rp))
        key[pos] = -1.0                                       # the true successor first inside its row
        o = np.lexsort((key, rp))
        start = np.r_[0, np.cumsum(np.bincount(rp, minlength=n))]
        rank = np.arange(len(o)) - start[rp[o]]
        npos = np.bincount(rp, weights=pos.astype(float), minlength=n).astype(np.int64)
        keep = o[rank < LK_NEG + npos[rp[o]]]                 # true successor (if a candidate) + 16 random negatives
        hits.append(((cand == true_loc[:, None]) & valid).any(1)[true_loc >= 0])
        SUB[int(s)] = dict(ii=ii, g=g, cand=cand, valid=valid, base=base, new=new[0],
                           Xtr=np.concatenate([base[keep].astype(np.float32), new[0][keep]], 1),
                           Ttr=pos[keep].astype(np.float32))
        print(f"sbj {s} fold {g} n={n} K={cand.shape[1]} pairs {valid.sum()} recall {hits[-1].mean():.3f} "
              f"[{time.time() - T_START:.0f}s] RSS {rss_gb()[0]:.1f} GB", flush=True)
        del cand, valid, base, new, rp, cj, pos, key, o
    del _V, _A
    print(f"scorer rows {sum(len(v['Ttr']) for v in SUB.values())} positive rate "
          f"{np.concatenate([v['Ttr'] for v in SUB.values()]).mean():.4f} candidate recall {np.concatenate(hits).mean():.4f}"
          f"   (notebook, all 22 subjects: 1,173,696 / 0.0549 / 0.930)")


def diag(succ, m_rows, tag):
    m = (succ >= 0) & m_rows
    ex = (succ[m] == TRUE_SUCC[m]).mean()
    has_t = m_rows & (TRUE_SUCC >= 0)
    rec = (succ[has_t] == TRUE_SUCC[has_t]).mean()
    sl = (d["y"][succ[m]] == d["y"][m]).mean()
    print(f"{tag}: linked {m.sum() / max(m_rows.sum(), 1):.3f} exact {ex:.3f} (of rows with a true successor: {rec:.3f}) "
          f"same-label {sl:.3f}", flush=True)
    return dict(linked=float(m.sum() / max(m_rows.sum(), 1)), exact=float(ex), exact_of_true=float(rec), same=float(sl))


with stage("combined links C-E: 5 fold scorers -> held-out fold links"):
    SUCC, SCORE = np.full(N_OOF, -1, np.int64), np.full(N_OOF, -50.0, np.float32)
    TOPC, TOPL = np.full((N_OOF, 3), -1, np.int64), np.full((N_OOF, 3), -50.0, np.float32)
    DIAG, IMP = {}, {}
    folds_present = sorted({v["g"] for v in SUB.values()})
    for g in folds_present:
        tr = [s for s, v in SUB.items() if v["g"] != g]
        if not tr:
            print(f"fold {g}: no training subjects (SMOKE), skipped")
            continue
        X = np.concatenate([SUB[s]["Xtr"] for s in tr])
        T = np.concatenate([SUB[s]["Ttr"] for s in tr])
        t0 = time.time()
        model = lgb.train(LK_PARAMS, lgb.Dataset(X, T, free_raw_data=True), num_boost_round=LK_ROUNDS)
        print(f"fold {g}: scorer on {len(tr)} subjects, {len(T)} rows (pos {T.mean():.4f}) in {time.time() - t0:.0f}s",
              flush=True)
        IMP[g] = model.feature_importance("gain")
        del X, T
        for s in [s for s, v in SUB.items() if v["g"] == g]:
            v = SUB[s]
            ii, cand, valid = v["ii"], v["cand"], v["valid"]
            lo = np.empty(valid.sum(), np.float32)
            for c0 in range(0, len(lo), 1_000_000):
                Xp = np.concatenate([v["base"][c0:c0 + 1_000_000].astype(np.float32), v["new"][c0:c0 + 1_000_000]], 1)
                p_ = np.clip(model.predict(Xp, num_threads=4), 1e-6, 1 - 1e-6)
                lo[c0:c0 + len(p_)] = np.log(p_ / (1 - p_))
                del Xp
            L = np.full(cand.shape, -50.0, np.float32)
            L[valid] = lo
            s_loc, sc = lk_assign(cand, L)
            SUCC[ii] = np.where(s_loc >= 0, ii[np.maximum(s_loc, 0)], -1)
            SCORE[ii] = sc
            o = np.argsort(-L, 1)[:, :3]
            cl = np.take_along_axis(cand, o, 1)
            ll = np.take_along_axis(L, o, 1)
            okk = (cl >= 0) & (ll > -50.0)
            TOPC[ii] = np.where(okk, ii[np.maximum(cl, 0)], -1)
            TOPL[ii] = np.where(okk, ll, -50.0)
            m = SUCC[ii] >= 0
            print(f"  sbj {s} n={len(ii)} K={cand.shape[1]} linked {m.mean():.3f} "
                  f"exact {(SUCC[ii][m] == TRUE_SUCC[ii][m]).mean():.3f} median score {np.median(sc[m]):.2f} "
                  f"[{time.time() - T_START:.0f}s]", flush=True)
            del L, lo
        del model
        gc.collect()
        DIAG[g] = diag(SUCC, d["fold"] == g, f"fold {g}")
    _m = SUCC >= 0
    assert len(np.unique(SUCC[_m])) == _m.sum() and (d["sbj"][SUCC[_m]] == d["sbj"][_m]).all()
    SCORE_QN = qnorm(SUCC, SCORE, d["sbj"], QN_REF)
    DIAG["all"] = diag(SUCC, np.ones(N_OOF, bool), "L2 OOF all folds")
    # top-1 candidate (before the 1:1 assignment)
    m1 = TOPC[:, 0] >= 0
    print(f"top-1 candidate: exact {(TOPC[m1, 0] == TRUE_SUCC[m1]).mean():.3f}")
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("raw linked score quantiles  ", np.round(np.quantile(SCORE[_m], qs), 3).tolist())
    print("QN_REF quantiles (notebook) ", np.round(np.quantile(QN_REF, qs), 3).tolist())
    for s in np.unique(d["sbj"]):
        mm = (d["sbj"] == s) & _m
        print(f"  sbj {s} fold {fold_of[int(s)]}: linked {mm.sum() / (d['sbj'] == s).sum():.3f} "
              f"exact {(SUCC[mm] == TRUE_SUCC[mm]).mean():.3f} raw median {np.median(SCORE[mm]):.2f}")
    # L0 links of the GPU fork, same rows, for reference
    l0 = np.load(KEEP / "links_L0.npz")
    if not SMOKE:
        DIAG["L0"] = diag(l0["oof_succ"].astype(np.int64), np.ones(N_OOF, bool), "L0 OOF (GPU fork) for reference")
    names = BASE49 + NEW16
    imp = np.mean([IMP[g] / IMP[g].sum() for g in IMP], 0)
    print("top features (gain share):", [(names[i], round(float(imp[i]), 3)) for i in np.argsort(-imp)[:15]])
    np.savez(WORK / "oof_L2.npz", succ=SUCC, score=SCORE, score_qn=SCORE_QN, top_cand=TOPC, top_lo=TOPL,
             true_succ=TRUE_SUCC, rows=OOF_ROWS, sbj=d["sbj"], fold=d["fold"], y=d["y"])
    (WORK / "oof_L2_diag.json").write_text(json.dumps(DIAG, indent=1))
    print("saved", WORK / "oof_L2.npz")

print("stage times (min):", {k: round(v / 60, 1) for k, v in TIMES.items()})
print(f"done in {(time.time() - T_START) / 60:.1f} min")
