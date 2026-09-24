# WEAR @HASCA 2026 — shared brief for experiment agents (2026-09-23)

Goal: public-LB macro-F1 >= 0.875 (top 3). Current best submission **e1 = 0.75001** (sim 0.7216 -> LB 0.750, i.e. LB ~= sim + 0.03).

## Hard rules
- NEVER kill python processes by name (no `Stop-Process -Name python`, no `taskkill /IM python.exe`). Other long jobs are running.
- Do NOT submit to Kaggle and do NOT create/push Kaggle notebooks/kernels. The lead does all submissions.
- Do NOT edit anything under `E:\Claude code\wear\src\` or `E:\Claude code\wear\data\` (read-only). Work only inside your own dir `E:\Claude code\wear\exp\<your_name>\` (copy code there if you need to modify it).
- CPU is shared (12 cores, 34 GB RAM, no GPU): cap LightGBM/numpy threads at 3 (`num_threads=3`, env `OMP_NUM_THREADS=3`).
- Python 3.11 with numpy, pandas, scipy, sklearn, lightgbm, xgboost, torch (CPU) available. Shell is Windows PowerShell 5.1 (no `&&`; avoid the word `rd(` in here-strings — PowerShell flags it as Remove-Item; prefer writing .py files and running them).

## Task structure (verified)
- 19 classes: 0 null, 1-5 jogging variants, 6-10 stretching variants, 11-12 push-ups (normal/complex), 13-14 sit-ups (n/c), 15 burpees, 16-17 lunges (n/c), 18 bench-dips. Metric: macro-F1 per window.
- Test: 12,234 windows = every 1-s tile of 4 unseen subjects' sessions (sbj 22: 5197, 23: 3128, 24: 1995, 25: 1914). ids are a uniform SHUFFLE (no temporal info). Each window: 50x3 acc of ONE random limb + the central 15 VideoMAE frame features.
- Train protocol: each subject does all 18 activities, each ~60-225 s total (median ~100 s) in 1-4 sets (segments), activity spans do not overlap, null in between. Per-subject null fraction ~= 1 - 1800/total_seconds (varies 15%-65%).

## Data (all local)
- `E:\Claude code\wear\data\prep\train_meta.csv` — 69,326 rows = seconds of 24 train sessions in temporal order: session, sbj, t, y (majority label), y_c, pur (label purity), n_nan_limbs.
- `train_imu.npy` (N,4,50,3) float16, limb order [left_arm, left_leg, right_arm, right_leg] (sbj_10 has NaN limbs 26%).
- `train_vid_pca.npy` (N,15,160) float16 = central-15 VideoMAE frames of each second projected by PCA-160 (fit on train frames; `pca_mean.npy`, `pca_components.npy`). `train_vid_mean768.npy` (N,768) raw mean over 15 frames.
- Test: `data\test\test_meta_data.csv` (id, sbj_id, sensor_location), `data\test\test_inertial_data.npy` (12234,50,3) float64, `data\prep\test_vid_pca.npy` (12234,15,160) same PCA, `data\prep\test_vid_mean768.npy`, raw `data\test\test_videomae_data.npy` (12234,768,15) float64.

## Current models (5-fold subject-disjoint; fold of subject s: `perm = np.random.RandomState(0).permutation(np.unique(meta.sbj)); fold = {s: i % 5 for i, s in enumerate(perm)}`)
- `work\lgbm_v1\oof.npy` (N,4,19) per (second, limb) OOF probs (NaN where limb missing or purity<0.8), `test.npy` (12234,19). Single-random-limb OOF macro-F1 0.630.
- `work\fusion_v1\oof.npy`, `test.npy` — Conv1D-IMU + Transformer-over-15x768-frames net (Kaggle GPU). OOF 0.566.
- Blend used for e1: log-space 0.8*lgbm + 0.2*fusion (OOF 0.650).
- `work\lgbm_v2\` (per-subject robust-normalised features) is being trained by the lead; use it if/when `test.npy` appears.

## Simulation harness (the validation that matters; it predicted the LB within 0.03)
- `src\chain.py` (pair features + LightGBM link scorer + linear assignment + cycle cut), `src\decode.py` (viterbi_chains, calibrate_counts, build_graph, graph_smooth), `src\simulate.py`, `src\sim_decode.py`, `src\predict.py` (test pipeline).
- `work\scorer.pkl` = link scorer fit on sessions sbj_1,3,7,12,16,19 (do not evaluate on those).
- `work\sim_struct.pkl` = dict session -> {a,b,n,y,limb,cand,lo,succ0,sc,Lm} for eval sessions sbj_0,5,10,14_2,20,21: rows a:b of train_meta, simulated random limb per second (`limb`), candidate successors `cand` (n,M) with scorer log-odds `lo`, linear-assignment successor `succ0`, edge score `sc`, dense matrix `Lm`.
- Per-window probs for a sim session: `P = oof[a:b][np.arange(n), limb]` (replace NaN rows by uniform).
- Best decoder so far (`sim_decode.py` column g_chain_cal_ps0.8): graph_smooth(P, build_graph(cand, lo, n, k=10), alpha=0.5, iters=5) -> chains = chains_from_succ(cut(succ0, sc, Lm, -6.0)) -> calibrate_counts(Pg, chains, lo=60, hi=160, p_stay=0.8).
  With the 0.8/0.2 blend, per session (sbj_0, 5, 10, 14_2, 20, 21): raw 0.515/0.638/0.513/0.620/0.628/0.601 (mean 0.586) -> decoded 0.641/0.725/0.579/0.773/0.798/0.814 (mean **0.7216**). Same decoder with the TRUE order (oracle chain): mean **0.8441**.
- Test-side structure: `work\test_structure.pkl` = dict sbj -> {idx (global ids), cand, lo, succ0, sc, Lm}.
- Link quality on sim (thr -6): exact-successor precision ~0.32, same-label edges ~0.87, true successor among candidates only 73-80%.

## Deliverables (every agent)
1. A markdown report `exp\<name>\REPORT.md` with the sim table (per session + mean, same 6 eval sessions; optionally extra sessions not in the scorer-fit list) comparing against the 0.7216 baseline.
2. A runnable script in your dir that writes a TEST submission CSV (`id,target_feature`, 12,234 rows, ids 0..12233) to `E:\Claude code\wear\subs\sub_<name>_<variant>.csv`, plus the per-window test probabilities you used (`.npy`).
3. Return the structured summary requested in your prompt. Numbers only from runs you actually executed.
