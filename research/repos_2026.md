# WEAR @ HASCA 2026 — public 2026 repos + why our pipeline plateaued

Date: 2026-09-23. Scope: (1) github.com/rilink/HARMA, (2) github.com/Agnuxo1/wear-hasca-2026, (3) the three public
kernels cached at `E:\Claude code\wear\acco\research\kernels\`, (4) our own `E:\Claude code\wear\acco\scripts\*.py`
+ `worklog.md`, (5) two read-only probes on the local test set. All numbers below are quoted from those sources or
measured here.

## 0. TL;DR

* HARMA (team at public LB 0.73709) is a **window-independent tabular stack**: 83 hand-crafted IMU features per
  (window, limb) row + pooled 32-d video PCA + a 64-d "temporal PCA" of the flattened 15x768 clip (fit on train+test,
  **mean-centred per subject incl. test subjects**) + a subject-adversarial LSTM (DANN) on the raw 15x768. Three
  LightGBMs -> Ridge/LGBM/GPBoost/TabM meta-learners -> max-confidence fusion. README: base models 0.591 / 0.663 /
  0.683, Ridge stack **0.725 CV / 0.732 LB**, final fusion **0.737 LB**. No temporal decoding, no test-structure use.
* Agnuxo1/wear-hasca-2026 is **not a submission**: one commit (2026-09-22 "Prepare public open-source release"),
  no model code, only a design doc (`PIPELINE_DESIGN.md`), a window-index builder, a schema inspector and a
  submission auditor. Its design doc is a reasonable plan (Conv1D IMU branch + 2-block Transformer over the 15
  frames + gated fusion + modality dropout, GroupKFold by subject), but nothing is trained or scored.
* Our pipeline is the same family as HARMA (per-limb LGBM + pooled video PCA + video-kNN histograms): row-F1 ~0.62
  OOF, LB 0.615 raw, 0.697 after blending public voters, 0.715 best-ever. HARMA's ceiling with far more stacking is
  0.737. **The whole "independent 1-s window classifier" family saturates at 0.70-0.74.** The 0.84-0.88 teams must be
  exploiting the test set's *structure* (each subject = every second of 2-3 sessions, each of the 18 activities is
  exactly one contiguous bout per subject) transductively, not a better window classifier.
* Two measured facts about the test set (this session, read-only): (a) **test ids are NOT in temporal order** —
  boundary-continuity of consecutive same-subject/same-limb windows equals random pairs (median 0.994 vs 1.080;
  intra-window step 0.166); (b) **VideoMAE embeddings are strongly subject/scene specific** — on 2,332 test windows
  the top-1 cosine neighbour is the same subject 99.7% of the time (same-subject top-1 median 0.913 vs cross-subject
  0.650), and there are **zero exact-duplicate video windows** (kills the public kernels' "4 limbs share one camera
  moment" hypothesis; one window per second, one random limb).

## 1. rilink/HARMA (public LB 0.73709)

Repo: https://github.com/rilink/HARMA — single commit 2026-07-08 "initial commit" by rilink (Ricarda H. Link,
Univ. Mannheim; the supplementary PDF is the feature appendix of Link & Stuckenschmidt, "Mitigating Null-Class
Dominance in Multiclass Inertial-Based Activity Recognition", UbiComp Companion '25 — i.e. their 2025 WEAR entry).

Files (bytes): README.md 4,426; feature_engineering/{build_feature_tables.py 9,394, build_video_sequences.py 4,826,
imu_features.py 17,098, HARMA-supp.pdf 462,210}; stacking/{adversarial_encoder.py 6,272, base_models.py 13,202,
combine_stacks.py 2,138, combine_stacks_alt_meta.py 7,813, cv_utils.py 4,618, meta_learner.py 3,929, oof_utils.py
2,776, run_ablation.py 3,400, run_adversarial.py 2,525, run_imu_adversarial.py 1,542, run_imu_only.py 1,630,
run_imu_temporal_pca.py 1,597, tabm.py 60,545}; requirements.txt (numpy pandas scipy scikit-learn lightgbm gpboost
torch rtdl_num_embeddings joblib).

### 1.1 Windowing / labels / alignment (feature_engineering/build_video_sequences.py, build_feature_tables.py)
* Train: 50-sample windows, **stride 25** over the raw 50 Hz CSV: `for start in range(0, len(inertial_df)-49, 25)`.
  Label = majority label of the 50 samples, window **dropped unless majority count > 40** (`if count <= 40: continue`).
  NaN label -> "null". 136,849 windows -> x4 limbs = 547,396 sensor rows ("each window appears 4 times").
* Video: `ratio = len(inertial)/video_len` (=5/3); `v_start=int(start/ratio)`, `v_end=int((start+50)/ratio)`
  (30 frames), then `_centered_window` keeps exactly 15 frames: `start = v_start + (30-15)//2` -> frames **+7..+21**.
  (Public kernels use +8..+22; the official test uses "central 15 of 30" — 1-frame ambiguity, both used at 0.67-0.74.)
* Test video read as (12234,768,15) and transposed to (N,15,768) — same as ours.
* One row per (window, sensor_location); `sensor_location` kept as a categorical LightGBM feature; the same video
  features are repeated on all 4 rows.

### 1.2 IMU features (imu_features.py + HARMA-supp.pdf) — "83 time/frequency-domain features per window"
Per axis (x,y,z), 18 each: mean, var, std, rms, shape_factor (=rms/|mean|), min, max, median, iqr, sma, skew,
kurtosis, jerk_mean, jerk_std, num_peaks (`find_peaks(distance=3)`), zero_crossings, FFT energy, dominant_freq,
FFT entropy, 10-bin signal entropy. Multi-axis (~40): double-integration position/velocity (dt=0.02, NOT gravity
corrected — no gyro), pitch/roll/tilt, total mean (sum & Euclidean), signed/abs signal magnitude, 3 axis
correlations + 3 covariances, SMV stats (Welch spectral entropy, min/max/ptp/iqr/std/skew/kurt), Hjorth mobility &
complexity, mean-crossing rate, differential entropy, Petrosian & Katz fractal dims. Some later removed (comment:
total, median, trajectory, ptp_smv, tilt, jerk_std). No augmentation anywhere.

### 1.3 Video features
* `build_feature_tables.py`: PCA(32) fit on a random 3,000 frames/subject of **train frames only**, then per window
  mean & std over the 15 centred frames of the 32 comps -> 64 "video_pca_*" columns (pooled; identical idea to ours).
* `base_models.py` "temporal PCA" model: flatten the clip `reshape(-1, 15*768)` (11,520-d), **IncrementalPCA(64)
  fitted once on all train+test windows combined**, then **per-subject mean-centring using train AND test sbj_id**:
  `for sbj in unique(all_subjs): all_pca[mask] -= all_pca[mask].mean(0)` -> columns `tpca_0..63`. This is the only
  transductive / domain-adaptive step in the repo and lifts the IMU-only LGBM from 0.591 to 0.663.
* `adversarial_encoder.py` (DANN): input (B,15,768) standardised with train mean/std over (windows, frames);
  1-layer unidirectional `nn.LSTM(768,128)`, **last hidden state** -> Linear->64 ReLU dropout 0.3 -> activity head (19)
  + subject head via gradient-reversal (domains = **train-fold subjects only**, test not used). 20-25 epochs, batch 128,
  Adam 1e-3, cosine schedule, lambda warm-up 0->1 (DANN sigmoid). No class weighting. Its 19-d softmax is appended as
  features to the IMU LGBM -> "Adversarial-IMU" 0.683.

### 1.4 Models, CV, stacking, post-processing
* Base LGBM (all three): `n_estimators=400, max_depth=15, learning_rate=0.1, subsample=0.9, colsample_bytree=1`.
  Null handling = **undersampling only**: `target_n = median(bincount(y[non-null])) * 3` null rows kept per fit
  (done after the subject split, train side only). No class weights, no null bias, no thresholds.
* CV: `partition_subjects` -> **6 subject-disjoint folds, seed 42** (22 subjects: 4 groups of 4 + 2 of 3);
  `sbj_0_2`/`sbj_14_2` grouped with their base subject. OOF probs (547,396 x 19) per base model; test preds from one
  model trained on all data.
* Meta: 3 x 19 OOF probs -> Ridge (alpha=1) / LGBM(200, depth 5, lr 0.05) / GPBoost(300, depth 6, lr 0.03) /
  TabM(K=8, 50 ep). `combine_stacks_alt_meta.py` fuses the three meta-learners by **max-confidence per row**.
  Ablation script (`run_ablation.py`) = leave-one-base-model-out with Ridge.
* README numbers: LightGBM-IMU 0.591, IMU-TempPCA 0.663, Adversarial-IMU 0.683 (OOF macro-F1); Ridge stack 0.725 CV
  / 0.732 LB; fusion 0.737 LB. CV tracks LB within ~1pt for this family — which also means their CV is honest and
  the gap to 0.88 is not a CV artefact.
* Nothing about temporal order, session structure, per-subject constraints, or smoothing. The Kaggle "team HARMA"
  at 0.73709 matches the README's 0.737.

## 2. Agnuxo1/wear-hasca-2026 (design doc only)

Repo: https://github.com/Agnuxo1/wear-hasca-2026 — one commit 2026-09-22T01:07Z "Prepare public open-source
release". Tree: README.md, PIPELINE_DESIGN.md (8.5 KB, Spanish/English), RULES.md, configs/baseline.yaml,
src/{build_train_index.py, audit_submission.py, inspect_schema.py}, 4 PNG figures, LICENSE (MIT). No training
code, no OOF, no scores. README itself: "an engineering baseline, not a clinical activity-recognition device".

Useful bits: `baseline.yaml` (stride 25, `min_label_purity 0.80`, `group_regex '^sbj_(\d+)'`, imu_dim 128,
video_dim 192, fusion_dim 128, modality_dropout 0.15, 80 epochs, lr 5e-4, bs 256). Design: Conv1D (k 5/5/3,
dil 1/2/4, GELU, BN, drop 0.15) + attention pooling + learned sensor-location embedding; video 768->192 LayerNorm +
2 Transformer blocks (4 heads, FFN 384) with CLS; gated fusion with per-modality "quality" scalars; CE with
label_smoothing 0.03 and class weights 1/sqrt(freq) clipped [0.5,3]; StratifiedGroupKFold; OOF-tuned temperature
and additive per-class biases; 3 seeds x 2 strides; softmax averaging. Warns "Kaggle announced VideoMAE feature
corrections" (re-download current files). It explicitly treats sbj_22-25 metadata as analysis-only. Nothing here
explains 0.88 either.

## 3. Public kernels cached locally (`wear\acco\research\kernels\`)

* `ts-emb-3wdc-temporal-fusion-ensemble.ipynb` (Nomannic, **0.61278 LB**): stride 25, windows inside label
  segments, video rows `start*3//5 + 8 .. +23`, 150 windows per (subject,class); PooledFusionModel (16 IMU stats +
  mean|std video 1536->192 + sensor emb) 3 epochs, TemporalFusionModel (Inception-1D IMU + 1-block Transformer over
  15 frames + gate + modality dropout 0.1, scale/noise aug) 4 epochs; blend 0.6/0.4; `probs[:,0] *= exp(0.75)`.
  Its EDA cell states: null = 39.7% of the train timeline; consecutive VideoMAE frames highly correlated.
* `0-670.ipynb` (akhyar, **0.670 LB**, "wearfusion" package embedded; modules extracted to scratchpad): champion =
  LightGBM on 100 FAME-style feats per limb (13 time + 8 freq + 4 sub-window, x 3 axes + magnitude) + video PCA(64)
  of mean|std + limb one-hot, inverse-frequency sample weights, `n_estimators 800, lr 0.07, num_leaves 63`, cap 600
  windows/(subject,class), `null_bias 0.75`; hierarchical family->variant variant (`run_hierarchical.py`) is what the
  cached notebook runs. Measured statements in its docstrings: "video classifier generalizes poorly (0.45)",
  "more training data hurts (0.657 -> 0.586)" (cap 600 beats using all windows — i.e. subject/recording domain shift
  dominates), the reference baseline reproduced at 0.61278. It also *assumed* "Sorting each subject's windows by
  id recovers its 1-second time axis" (`temporal_smooth.py`) — **false**, see 5(a).
* `wear-hasca.ipynb` ("V8_4IMU_VIDEO_JOIN_VITERBI", prior score 0.639): trains a 4-limb LightGBM with sensor-dropout
  copies (keep 1-3 limbs, p=.30/.35/.35), tries to join test windows that share the same camera moment via exact
  video hash / rounded fingerprint / greedy cosine >= 0.98, rebuild (50,4,3), then Viterbi (stay 0.93) over
  min-id order. Both the join (no duplicate video windows exist) and the id-order Viterbi are built on false premises.

## 4. Our previous pipeline (`wear\acco\scripts`) and why it plateaued at 0.62-0.72

What it does (`build_features_v9.py`, `train_v9.py`, `decode_v9*.py`, `label_prop.py`, `make_pseudo.py`):
* Windows: stride 25, label = `y[::25]` (**label of the first sample only**, no purity filter — boundary windows are
  mislabeled; HARMA drops anything <80% pure, akhyar windows inside segments only).
* 105 IMU feats per limb row (16 stats + 8 spectral per axis and magnitude, smoothed/residual, 3 correlations);
  video = **pooled mean/std only**, from `videomae_pooled/*_mean|std.npy` that were computed on fixed 15-frame chunks
  and re-mixed 7/8 across chunks (an approximation of the true central frames — `proc_video.py` deleted the raw
  frames, so the 15x768 temporal structure was never available locally); TruncatedSVD(48) on train mean|std;
  fold-safe video-kNN label histograms (K=15, 40 feats).
* 4-fold subject-disjoint LightGBM (lr 0.12, 47 leaves, 200 rounds, min_data 200, **trained on only even windows**
  `winof % 2 == 0`), rounds picked on a 30k subsample; test = logit-mean of the 4 fold models.
* Decode: video-kNN propagation among test windows (k=12, alpha .25, 2 iters), null-logit override toward the
  video-kNN null mass, per-subject bisection so predicted null rate == kNN prior null rate; self-training with
  pseudo weight 0.35; blends with public voters (aka x0.8, udaken x0.25).
* Results (worklog + download/README): row-F1 ~0.62 OOF; d1 0.61462, d2 0.61841, d3 (+aka votes) 0.69065, d5
  0.68418, d6 0.65535, d7 0.69707; best ever v8 0.71528.

Diagnosis:
1. **Same model family as HARMA, fewer tricks.** HARMA's plain IMU LGBM is 0.591 OOF; ours ~0.62 row-level; HARMA
   needs temporal-PCA + per-subject centring + DANN + 4 meta-learners to reach 0.737. Our +7pt from blending public
   kernels is the same "average several 0.6-0.7 window classifiers" effect. There is no evidence anywhere that this
   family exceeds ~0.75.
2. **Video-kNN on test retrieves the wrong thing.** Test video neighbours are overwhelmingly same-subject (99.7%
   top-1) and train contains none of the test subjects, so (i) the train-kNN label histograms are cross-subject
   lookups that generalise poorly (akhyar measured a video-only classifier at 0.45 LB), and (ii) the kNN null
   prior systematically underestimated test null (worklog: 38% vs optimum ~45%) — the per-subject null-rate matching
   in `decode_v9c.py` then inherited that bias (d6 overshoot).
3. **Pooled video throws away the only within-window temporal signal**, but that is worth ~+2pt (HARMA: pooled
   PCA 0.663 vs LSTM-DANN 0.683), not the missing 15pt.
4. **Label noise + data-hungry training.** First-sample labels, no purity filter, half the windows dropped, 200
   rounds at lr 0.12 with min_data_in_leaf 200 — a weaker base than akhyar's 800-round champion (0.657-0.670).
5. **"More data hurts" was never addressed.** akhyar: 0.657 -> 0.586 when uncapped; HARMA undersamples null to
   3x the median class; we used everything with weight 1. Recording/subject domain shift is the dominant error
   source (test subjects are new *people* at new *locations* 12-15), and only HARMA's per-subject mean-centring
   attacks it — and that step alone was worth +7pt for them.
6. **Post-processing was tuned to OOF with train subjects as the pool**, but the test pool is 4 subjects x thousands
   of same-scene windows; propagation there mainly copies same-subject predictions around (a different regime than
   the 22-subject OOF pool), so OOF gains (+5pt) did not transfer.

## 5. Test-structure facts measured here (read-only, `scratchpad\probe_test.py`, `probe_video.py`)

(a) `test_meta_data.csv` ids are shuffled across subjects (fraction of consecutive ids with equal sbj 0.2942 vs
    random expectation 0.2969) **and within subject the id order is not temporal**: for the 916 consecutive-id
    pairs with same subject and same limb, ||x_{i+1}[0] - x_i[-1]|| median 0.994 vs 1.080 for random same-subject
    same-limb pairs, while the intra-window sample step is 0.166. Any id-order smoothing/Viterbi is noise.
(b) Sensor counts: left_leg 3068, right_leg 3060, left_arm 3058, right_arm 3048; subjects 22:5197, 23:3128,
    24:1995, 25:1914 = session durations in seconds -> exactly one window per second, one random limb.
(c) Zero exact-duplicate VideoMAE windows among the first 2,332 (no shared "camera moments").
(d) Video embeddings (mean|std, L2) identify the subject/scene: same-subject top-1 cosine p10/p50/p90 = 0.839 /
    0.913 / 0.961, cross-subject 0.505 / 0.650 / 0.767 — on only ~19% of each subject's windows; with all windows
    the same-subject neighbours will be even tighter. Mean adjacent-frame cosine inside a window 0.962.
(e) Local data status: `E:\Claude code\wear\data\train` does not exist yet and `test_videomae_data.npy` was still
    growing during this session (215 MB -> 334 MB of 1.13 GB; header says (12234,768,15) float64). Do not build
    features from it until it is complete (the downloader is a background python process — never kill by name).

## 6. What the 0.84-0.88 teams are most likely doing (inference, ranked)

1. **Transductive per-subject segmentation in video space + constrained assignment.** Each test subject's windows
   are one contiguous timeline where each of the 18 activities is a single bout separated by null; VideoMAE
   features cluster by scene/pose/time (5d). Cluster each subject's windows (video kNN graph / spectral / HDBSCAN,
   optionally chained into an ordering via nearest-neighbour Hamiltonian path), aggregate IMU-model log-probs per
   cluster, then assign labels with the constraint "each non-null class at most one bout per subject" (Hungarian
   on cluster x class cost) and everything else null. Bout-level evidence (hundreds of windows) is far stronger
   than 1-s windows; this is the only mechanism that plausibly turns 0.72 into 0.88 with 4-6 submissions.
2. **Per-subject feature adaptation** (HARMA's centring generalised: per-subject standardisation of IMU features and
   video PCA, or DANN with test subjects as unlabeled domains) — cheap, +5-7pt in HARMA's own ablation.
3. Temporal video model on the raw 15x768 (Transformer/LSTM) + sensor-dropout IMU CNN, stacked — worth a few points
   on top, not the gap.

## 7. Concrete next steps for our repo
* Rebuild features with 80% label purity, windows inside label segments, central frames +7/+8..+22 kept as 15x768.
* Add HARMA's temporal PCA(64) with per-subject centring (test subjects included) and akhyar's 800-round LGBM with
  1/freq weights, cap 600/(subject,class), null undersampled to 3x median.
* Validate transductive bout decoding on train subjects (hold out 4, tile every second, random limb, shuffle) —
  this simulation is exactly reproducible from the train CSVs and measures the real upside before spending
  submissions.
