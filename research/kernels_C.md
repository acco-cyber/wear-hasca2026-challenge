# Kernel analysis C: lakhindarpal / hmnshudhmn24 / abhinavm2811 / stmugiwara

Competition: `3rd-wear-dataset-challenge-hasca-2026` (macro F1, 19 classes). Analysed 2026-09-22 from the local kernel copies under `E:\Claude code\wear\kernels\`. Every code and output cell was read (notebooks were dumped cell-by-cell with a script; the lakhindarpal notebook has ZERO saved outputs, the abhinavm2811 notebook has full outputs, the other two have no outputs).

Headline: none of these four kernels documents anything close to the 0.84-0.88 public LB band. The best measured number in this batch is abhinavm2811's subject-independent OOF macro F1 = 0.5871 (0.6019 after per-class prior calibration). All four use only POOLED video (mean or mean+std over frames) and hand-made IMU statistics; none uses the 15x768 temporal video structure, none exploits test-set structure, none does any temporal smoothing. The 0.88 recipes are therefore not in this batch.

---

## 1. lakhindarpal / "3rd WEAR Dataset Challenge @HASCA 2026" (id_no 115877190, GPU T4)

File: `E:\Claude code\wear\kernels\lakhindarpal__3rd-wear-dataset-challenge-hasca-2026\3rd-wear-dataset-challenge-hasca-2026.ipynb` (27 cells, no outputs saved).

- **Model**: CatBoostClassifier on GPU, `loss_function=MultiClass`, `eval_metric=TotalF1:average=Macro`, iterations 3000, lr 0.04778, depth 4, l2_leaf_reg 3.71, random_strength 3.127, bagging_temperature 0.267, od_wait 100 + early_stopping_rounds 50, `use_best_model=True`. Hyper-parameters look like an Optuna result pasted in; the tuning code is not in the notebook.
- **Windowing (train)**: 50-sample (1 s) windows with **stride 25 (50% overlap)** over each CSV; label = `mode()` of the 50 labels; `label.fillna("null")`.
- **Video alignment**: `start_f = start_idx/50*30`, `center_f = start_f + 15`, uses frames `[center_f-7, center_f+8)` = 15 frames, i.e. the **central 15 of the 30 frames of the second**, exactly the test protocol. Then `np.mean(axis=0)` -> 768-d pooled vector (`vi_0..vi_767`). No std pooling, no sequence use.
- **Inertial features**: per limb, mean/std/min/max per axis = 12 features (`in_0..in_11`). Each window produces **4 rows, one per limb**, each with `sensor_location` as a CatBoost categorical feature (matches the single-limb test format).
- **Feature selection**: a 200-iteration depth-4 CatBoost probe on all 780 features, keep top 300 by importance (+ `sensor_location`).
- **CV**: `GroupKFold(n_splits=5)` grouped by file name (`sbj_0` and `sbj_0_2` are DIFFERENT groups -> slight subject leakage between the repeated-session files). Fold F1 printed but **no outputs are stored in the notebook**, so no CV number is available. Test inference = sum of the 5 fold models' `predict_proba`, argmax.
- **Null handling / post-processing**: none. No class weights, no prior correction, no smoothing.
- **Test handling**: `vid_p = np.mean(test_vid[i], axis=1)` on a (768,15) slice -> correct mean over frames. Location column auto-detected among `sensor_location / inertial_sensor_location / location`.
- **Test-set structure**: nothing observed.
- Verdict: clean, correct baseline design (limb-rows + categorical location + central-15-frame mean). Expected LB in the 0.6-0.7 band like ours; nothing that explains 0.88.

## 2. hmnshudhmn24 / "3rd WEAR Dataset Challenge @HASCA 2026" (id_no 115550182)

File: `E:\Claude code\wear\kernels\hmnshudhmn24__3rd-wear-dataset-challenge-hasca-2026\3rd-wear-dataset-challenge-hasca-2026.ipynb` (2 cells, no outputs).

- **Model**: `RandomForestClassifier(n_estimators=100)`, no CV at all.
- **Windowing**: non-overlapping 1 s windows (`i*50:(i+1)*50`), label = mode; `num_windows = min(len(df)//50, len(video)//30)`.
- **Video**: frames `i*30+7 : i*30+22` = central 15 frames, `mean(axis=0)` -> 768-d.
- **Inertial**: **only `right_arm_acc_{x,y,z}` mean** (3 numbers). Trains on right arm only although test windows come from all four limbs; the location column of test meta is never read.
- **Feature vector**: 3 + 768 = 771 dims; essentially a video-mean classifier.
- **Test**: mean over the sample axis of inertial and over the frame axis of video (axis auto-detected: `vid_axis = 1 if shape[1]==15 else 2`, so the (12234,768,15) layout is handled). Predictions written as float.
- **Null / post-processing / CV / metrics**: none of any kind. Nothing about test structure.
- Verdict: trivial baseline; no usable numbers.

## 3. abhinavm2811 / "3rd WEAR Dataset Challenge @HASCA 2026" (id_no 134245040, CPU, full outputs)

File: `E:\Claude code\wear\kernels\abhinavm2811__3rd-wear-dataset-challenge-hasca-2026\3rd-wear-dataset-challenge-hasca-2026.ipynb` (4 cells; cell 2 = pipeline with output, cell 3 = calibration with output).

### Pipeline (cell 2, "Fast Vectorized Pipeline")
- **Model**: LightGBM multiclass, lr 0.05, num_leaves 63, subsample 0.8, colsample_bytree 0.7, reg_lambda 1, max_bin 63, max 600 rounds, early stopping 50 on validation multi_logloss. `compute_class_weight("balanced")` sample weights.
- **Windowing**: 1 s windows, **stride 50 (no overlap)**, label = label at the window CENTER sample (not mode). Per file 4 rows per window (one per limb) with a 4-d location one-hot. Total **277,304 rows = 69,326 windows x 4**, feature dim 1592.
- **Inertial features (52 + 4 one-hot)**: per axis mean, std, min, max, median, skew, kurtosis, RMS, sum|diff|, p25, p75, 4 rfft band sums (15 x 3 = 45); magnitude mean/std/max/RMS (4); pairwise axis correlations (3).
- **Video**: `gather_video_windows` picks 15 frames **linearly spaced from f_start to f_end of the whole second (frames 0..30)**, NOT the central 15 -> mean+std pooling (1536-d). This is a train/test mismatch: test windows are the central 15 contiguous frames, so the test std over frames is systematically smaller than in training.
- **CV**: `GroupKFold(n_splits=3)` grouped by the `sbj_id` COLUMN (so `sbj_0` and `sbj_0_2` share a group -> proper subject-level split).
- **Printed metrics** (subject-independent OOF):
  - Fold 0: 0.5805 (best_iter 55), Fold 1: 0.5930 (54), Fold 2: 0.5841 (59). **OOF macro F1 = 0.5871**, accuracy 0.650. Each fold took ~20 min on CPU; total 62.8 min.
  - Best iterations of ~55 at lr 0.05 = the model overfits subjects almost immediately (the 1536 video dims dominate).
  - Per-class OOF F1: null 0.731 (P 0.693 / R 0.773, support 110,140); jogging 0.746; jogging rotating-arms 0.669; skipping 0.636; sidesteps 0.735; butt-kicks 0.682; **stretching triceps 0.389, lunging 0.334, shoulders 0.309** (the three worst); hamstrings 0.709; lumbar rotation 0.774; push-ups 0.535; push-ups complex 0.504; sit-ups 0.631; sit-ups complex 0.595; burpees 0.589; lunges 0.475; lunges complex 0.626; bench-dips 0.485 (recall only 0.381).
  - Train class support (rows /4 = windows): null 110,140 rows = 27,535 windows = **39.7 % of training seconds are null**; every activity class has 8.4k-10k rows = ~2.1k-2.5k windows across 24 files -> **~90-105 s per activity bout per session**.
- **Test-set facts printed**: `test_inertial_data shape: (12234, 50, 3)`, `test_videomae_data shape (as loaded): (12234, 768, 15)` -> transposed to (12234,15,768); location values `{'left_leg','right_leg','left_arm','right_arm'}`.
- **Predicted test distribution (uncalibrated)**: null 7210 / 12234 = **58.9 %**; smallest classes: stretching triceps 80, push-ups 73, lunging 123. Compare the train null rate 39.7 % -> the model over-predicts null on test (or test genuinely has more null; see open questions).

### Post-processing (cell 3, "Post-hoc per-class probability calibration")
- Coordinate-ascent search of per-class probability multipliers on the grid 0.10..2.00 step 0.05, 4 passes, objective = OOF macro F1, then apply the same multipliers to averaged test probs.
- Result: OOF 0.5871 -> 0.6004 -> 0.6014 -> 0.6017 -> **0.6019 (+0.0148)**. Final weights `[0.75 0.85 1.0 0.8 0.55 0.75 1.55 1.5 1.25 0.5 1.2 2.0 1.45 1.25 0.95 0.75 0.8 0.8 1.95]` (null 0.75; push-ups 2.0; bench-dips 1.95; stretching triceps/lunging 1.55/1.5).
- Test null count after calibration 7210 -> **6324 (51.7 %)**; stretching shoulders 246 -> 600, push-ups 73 -> 159, bench-dips 196 -> 311.
- Nothing temporal, no test-subject-level constraints. No LB number is recorded in the notebook.

## 4. stmugiwara / "WEAR v7 Full5" (id_no 132380236)

File: `E:\Claude code\wear\kernels\stmugiwara__wear-v7-full5\wear-v7-full5.ipynb` (1 code cell, no outputs).

- Reads only the FIRST 5 train CSVs, concatenates raw 50 Hz rows (no windowing), `dropna(subset=[label])` (**drops all null rows if null is stored as NaN**, so the model can never predict null), `LabelEncoder` (alphabetical class ids, NOT the competition 0-18 mapping), numeric columns = `sbj_id` + all 12 acc channels, 20k random rows, StandardScaler, `RandomForestClassifier(200, max_depth=20)`.
- Test: mean over the 50 samples of the 3-axis window -> 3 numbers padded with zeros to 13 columns (so the test limb always lands on the `sbj_id`/`right_arm` columns regardless of `sensor_location`). No video at all.
- No CV, no metrics, no null handling, no post-processing, nothing on test structure. This is a broken smoke-test kernel and should be ignored.

---

## Cross-kernel comparison

| kernel | model | IMU feats | video | train stride | rows/window | CV | OOF macro F1 | post-proc |
|---|---|---|---|---|---|---|---|---|
| lakhindarpal | CatBoost GPU, top-300 feats | 12 stats | central-15 mean (768) | 25 | 4 (limb rows, cat. loc) | 5-fold GroupKFold by file | not saved | none |
| hmnshudhmn24 | RF-100 | right-arm mean (3) | central-15 mean (768) | 50 | 1 (right arm only) | none | none | none |
| abhinavm2811 | LightGBM, balanced weights | 52 stats + loc one-hot | 15 frames spanning the full second, mean+std (1536) | 50 | 4 | 3-fold GroupKFold by sbj_id | **0.5871 -> 0.6019** calibrated | per-class prob multipliers |
| stmugiwara | RF-200 on raw samples | raw 12 ch | none | none | n/a | none | none | none |

## What this batch implies for a 0.85+ recipe

1. Pooled-video + IMU-stat GBDTs plateau at OOF ~0.59-0.60 and LB ~0.6-0.72 (consistent with our own 0.715). All three "real" kernels here are that recipe; the 0.84-0.88 LB entries must be doing something categorically different (temporal modelling over the 15x768 frames, subject-transductive/kNN methods on the test set, or exploiting the fact that each test activity is one contiguous bout per subject).
2. The per-class OOF profile is consistent across kernels: stretching (triceps/lunging/shoulders) and bench-dips/push-ups are where macro F1 is lost; these are quasi-static postures where the IMU stats of a single limb are nearly uninformative and video temporal structure should matter most.
3. Per-class prior calibration is worth only ~+0.015 OOF; null over-prediction on test (59 % predicted null vs 40 % null in training) is a bigger structural issue that a per-subject null-rate constraint (18 bouts x ~95 s per subject = ~1700 activity seconds; e.g. sbj_23 has 3128 windows -> ~45 % null, sbj_22 has 5197 -> ~67 % null) could address directly.
4. The abhinavm2811 kernel's frame gather (linspace over the full 30-frame second) is a subtle train/test mismatch to avoid: always take frames 7..21 of each second to match the test's central-15 protocol (lakhindarpal and hmnshudhmn24 do this correctly).
5. Group the repeated-session files (`sbj_0_2`, `sbj_14_2`) with their subject in CV (abhinavm2811 does, lakhindarpal does not).
