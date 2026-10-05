"""Short GPU job (~10 min on T4x2): two extra fusion window-model seeds for the big-pool-opt-s7 fit. Keeps only the
setup / preprocessing / test-like tiles / window-training cells of the v4 notebook, trains two new fusion seeds (one
per GPU), then rebuilds s7's window blend with them (8 + 2 fusion seeds, s7's 4 IMU seeds; s7's per-run outputs are
read from its kernel output) and saves the new window log-probs for the local decode.
  python patch_wseed.py <tag> <seedA> <seedB>    -> kaggle/v4_fork/wseed-<tag>/"""
import json, os, sys
W = r"E:\Claude code\wear"
SRC = os.path.join(W, "kaggle", "pub_1004", "good", "wear-hasca-learned-links-counts-lb-0-92942.ipynb")
KEEP_CELLS = (6, 8, 9, 10, 11, 12)
RUNS_OLD = '''FULL_RUNS = [  # (name, args) in priority order; --knn/--eval_every only change logging
    ("final_s0", "--seed 0 --epochs 12 --eval_every 4 --knn 1,3,5,7,10"),
    ("imu_s0", "--seed 0 --epochs 30 --eval_every 5 --only imu --knn 1,3,5"),
    ("final_s1", "--seed 1 --epochs 12 --eval_every 4 --knn 1,3,5,7,10"),
    ("final_s2", "--seed 2 --epochs 12 --eval_every 4 --knn 1,3,5"),
    ("final_s3", "--seed 3 --epochs 12 --eval_every 4 --knn 1,3,5"),
    ("imu_s1", "--seed 1 --epochs 30 --eval_every 5 --only imu --knn 1,3,5"),
]'''
FINAL = r'''
# ---- FORK: s7's window blend rebuilt with the two new fusion seeds
import glob
_old = sorted(glob.glob("/kaggle/input/**/keep4/runs/*.npz", recursive=True))
print("s7 runs found:", [os.path.basename(p) for p in _old])
_fo = [p for p in _old if os.path.basename(p).startswith("final_s")]
_io = [p for p in _old if os.path.basename(p).startswith("imu_s")]
assert len(_fo) == 8 and len(_io) == 4, (_fo, _io)
_rn = lambda x: x - np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)) - x.max(1, keepdims=True)
_s7 = sorted(glob.glob("/kaggle/input/**/keep4/stage.npz", recursive=True))[0]
_st = np.load(_s7, allow_pickle=True)
_new_o = [np.load(r / "oof_raw.npz")["logp"].astype(np.float32) for r in FUSION]
_new_t = [_rn(np.load(r / "test_logp_raw.npy").astype(np.float32)) for r in FUSION]
assert (np.load(FUSION[0] / "oof_raw.npz")["y"] == _st["oof_y"]).all()
_fo_o = [np.load(p)["oof_logp"].astype(np.float32) for p in _fo]; _fo_t = [_rn(np.load(p)["test_logp"].astype(np.float32)) for p in _fo]
_io_o = [np.load(p)["oof_logp"].astype(np.float32) for p in _io]; _io_t = [_rn(np.load(p)["test_logp"].astype(np.float32)) for p in _io]
_chk = blend(mean_logp(_fo_o), mean_logp(_io_o), CFG["imu_w"])
print(f"s7 window blend rebuilt from its runs: max abs diff {np.abs(_chk - _st['oof_logp']).max():.4f}")
_prev_o, _prev_t = [], []
for _p in sorted(glob.glob("/kaggle/input/**/wseed/window_logp.npz", recursive=True)):   # seeds of earlier short jobs
    _z = np.load(_p); _prev_o += [x.astype(np.float32) for x in _z["new_oof"]]; _prev_t += [x.astype(np.float32) for x in _z["new_test"]]
print("earlier extra seeds:", len(_prev_o))
_lo = blend(mean_logp(_fo_o + _prev_o + _new_o), mean_logp(_io_o), CFG["imu_w"])
_lt = blend(mean_logp(_fo_t + _prev_t + _new_t), mean_logp(_io_t), CFG["imu_w"])
print(f"window OOF macro F1: s7 {macro_f1(_st['oof_y'], _st['oof_logp'].argmax(1)):.4f} -> with new seeds {macro_f1(_st['oof_y'], _lo.argmax(1)):.4f}")
_out = WORK / "wseed"; _out.mkdir(exist_ok=True)
np.savez(_out / "window_logp.npz", oof_logp=_lo.astype(np.float32), test_logp=_lt.astype(np.float32),
         new_oof=np.stack(_new_o).astype(np.float16), new_test=np.stack(_new_t).astype(np.float16))
print("saved", _out / "window_logp.npz")
'''


def build(tag, sa, sb, arch="pool", prev=()):
    nb = json.load(open(SRC, encoding="utf-8"))
    xa = " --vid_arch tf" if arch == "tf" else ""
    cells = [c for i, c in enumerate(nb["cells"]) if i in KEEP_CELLS]
    n = dict(runs=0, ess=0, asrt=0)
    for c in cells:
        s = "".join(c["source"])
        if RUNS_OLD in s:
            s = s.replace(RUNS_OLD, "FULL_RUNS = [   # FORK: two extra fusion seeds, one per GPU\n"
                                    f'    ("final_s0", "--seed {sa} --epochs 12 --eval_every 4 --knn 1,3,5{xa}"),\n'
                                    f'    ("final_s1", "--seed {sb} --epochs 12 --eval_every 4 --knn 1,3,5{xa}"),\n]'); n["runs"] += 1
        if 'ESSENTIAL = ("final_s0", "imu_s0")' in s:
            s = s.replace('ESSENTIAL = ("final_s0", "imu_s0")', 'ESSENTIAL = ("final_s0",)   # FORK'); n["ess"] += 1
        if 'assert FUSION and FUSION[0].name == "final_s0" and IMU, (FUSION, IMU)' in s:
            s = s.replace('assert FUSION and FUSION[0].name == "final_s0" and IMU, (FUSION, IMU)', 'assert len(FUSION) == 2, FUSION   # FORK'); n["asrt"] += 1
        c["source"] = s.splitlines(keepends=True)
    assert all(v == 1 for v in n.values()), n
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": FINAL.splitlines(keepends=True)})
    nb["cells"] = cells
    out = os.path.join(W, "kaggle", "v4_fork", f"wseed-{tag}"); os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, f"wseed_{tag}.ipynb"), "w", encoding="utf-8"))
    json.dump({"id": f"koushikrudra/wear-wseed-{tag}", "title": f"wear-wseed-{tag}", "code_file": f"wseed_{tag}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
               "dataset_sources": [], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"],
               "kernel_sources": ["koushikrudra/wear-v4-big-pool-opt-s7"] + [f"koushikrudra/wear-wseed-{p}" for p in prev],
               "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
              open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    print("built", out, n)


if __name__ == "__main__":
    # python patch_wseed.py <tag> <seedA> <seedB> [pool|tf] [prev_tag,prev_tag]
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "pool",
          tuple(p for p in (sys.argv[5].split(",") if len(sys.argv) > 5 else []) if p))
