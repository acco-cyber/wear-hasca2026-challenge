# Kernel analysis A: akhyar2612 "0.670" and udaken10 "Base_line_liner_model"

Competition: 3rd WEAR Dataset Challenge @HASCA 2026 (macro-F1, 19 classes). Analysed 2026-09-22.

Sources read in full:
- `E:\Claude code\wear\kernels\akhyar2612__0-670\0-670.ipynb` (kernel id akhyar2612/0-670, id_no 133773420, CPU-only, no internet)
- `E:\Claude code\wear\kernels\udaken10__base-line-liner-model\base-line-liner-model.ipynb` (kernel id udaken10/base-line-liner-model, id_no 131761550, CPU-only)
- Extracted akhyar modules (29 files) at `C:\Users\Koushik\AppData\Local\Temp\claude\E--Claude-code\4892a651-709e-47dc-8a3d-bbb791e079fe\scratchpad\akhyar\cell2_*.py`
- Cross-checks: `E:\Claude code\wear\data\test\test_meta_data.csv`, `E:\Claude code\wear\data\participant_meta_data.txt`, `E:\Claude code\wear\acco\worklog.md`

Important caveat on "printed metrics": **neither notebook has any saved outputs** (all code cells: `outputs=[]`, `execution_count=None`). Every number below comes from code, docstrings and CLI defaults, plus our own worklog. Also note: the train data (`E:\Claude code\wear\data\train\...`) is **not present on this machine** at the time of writing (only `test/`, `sample_submission.csv`, `participant_meta_data.txt` exist), so training-set sizes are analytic estimates cross-checked against our worklog, not re-measured.

---

## 1. akhyar2612 / "0.670" (public LB 0.670)

### 1.1 What the notebook physically is
- 2 code cells. Each cell writes a 29-module Python package `wearfusion` into `/kaggle/working/wearfusion/` from an embedded `SRC` dict (cell 1 and cell 2 embed **byte-identical** modules; verified by diffing all 29 extracted files). Cell 2 then runs the only entry point that matters:
  ```python
  sys.argv = ["run_hierarchical", "--stride", "25", "--max-per-class", "600",
              "--n-estimators", "800", "--lr", "0.07", "--num-leaves", "63",
              "--video-weight", "0.3", "--null-bias", "0.75"]
  hier_main()   # comment in cell: "Run on CPU (~3-4h). Writes submission.csv."
  ```
- The package is the residue of a long private campaign: a neural fusion net (`models.py`, `train.py`), a port of Nomannic's "TS/Emb 3WDC Temporal Fusion Ensemble" (`reference_baseline.py`, docstring: "scored 0.61278"), FAME-style multi-view net, a "strong" fusion net, a video-only transformer, video kNN, sensor-specialist LGBMs, domain-weighting, recording selection, cross-sensor propagation, temporal smoothing, and the hierarchical LightGBM that was finally submitted. Only `run_hierarchical` -> `feature_data` -> `features` -> `data` -> `inference.write_submission` is on the executed path.

### 1.2 Train windowing (`data.build_windows`, `subsample_windows`)
- Window = 50 inertial samples (1 s @ 50 Hz), **stride 25 samples (0.5 s)**.
- Windows are generated **inside contiguous label segments only**: segment boundaries from `labels[1:] != labels[:-1]`; the first window start is `ceil(seg_start/25)*25`; windows run to `seg_end-50`. A window therefore **never straddles two labels**, and segments shorter than 50 samples are dropped. Label = the segment label (NaN/unknown label -> 0 = null).
- Video rows for a window starting at inertial sample `i`: `v0 = i*3//5 + 8`, frames `v0 .. v0+14` (15 frames) — i.e. the **central 15 of the 30 frames** of that second (`VIDEO_FRAME_OFFSET = 8`). This matches the test's "central 15 of 30" and our worklog's independently derived "+8..+22".
- Cap: `--max-per-class 600` = **max 600 windows per (sbj_id, class)** (keyed on the numeric subject id, so `sbj_0` and `sbj_0_2` share one cap), chosen randomly with seed 42. In practice this mainly caps null (each subject has thousands of null windows at stride 0.5 s; activity bouts give ~100-250 windows each).
- Each window becomes **4 training rows** ("views"): one per limb, each with that limb's 100 inertial features, the same 64 video features, and a 4-d one-hot of the limb. `y` is repeated 4x. **No `valid`-sensor masking on this path**: `np.nan_to_num` turns a missing sensor into an all-zero signal that still carries the activity label (the neural path in `train.py` masks NaN sensors via `valid`; the boosting path does not).
- Size estimate: our worklog measured 138,632 windows / 554,528 limb rows at stride 25 uncapped; with the 600 cap null collapses to <= 600 x 22 subjects = 13,200 windows, so the training matrix is roughly 60-90k windows x 4 = **~250-350k rows x 168 features** (estimate; train data not on disk here).

### 1.3 Features (168 per row) — `features.py`, `feature_data.py`
- **Inertial, 100-d per limb** = 4 streams (x, y, z, |a|) x 25 features:
  - 13 time: mean, std, rms, min, max, ptp, median, skew, kurtosis, zero-crossing rate, mean-crossing rate, lag-1 autocorr, jerk (std of first difference)
  - 8 frequency (rfft of zero-meaned signal, 50 Hz): dominant freq, spectral centroid, spectral entropy, band powers <2 Hz / 2-8 Hz / >=8 Hz, 95% rolloff, spectral flux
  - 4 sub-window: mean/std difference between halves, energy ratio, linear slope
  - No standardisation (trees). Fully vectorised numpy; seconds for the whole set.
- A 159-d **orientation-invariant block** (gravity/body decomposition, rotation-invariant products, covariance eigenvalues) exists (`extract_orientation_invariant`, flag `use_orientation`) but is **OFF** in the submitted run.
- **Video, 64-d**: `extract_video_features_vect` = concat[mean over 15 frames, std over 15 frames] (1536-d) @ a **fixed Gaussian random projection** (`RandomState(0)`, 1536 -> 64, scaled 1/sqrt(1536)). **Not PCA, not learned** (`run_boost` has a PCA option; `run_hierarchical` passes `video_transform=None`). The temporal order of the 15 frames is discarded; only mean and std survive.
- **Sensor one-hot, 4-d.**
- Test: `test_inertial_data.npy` (N,50,3) -> same 100-d block; `test_videomae_data.npy` (N,768,15) transposed to (N,15,768) -> same 64-d projection; sensor one-hot from `test_meta_data.csv.sensor_location` (verified the values are exactly the 4 strings `right_arm/right_leg/left_leg/left_arm`, counts 3048/3060/3068/3058).

### 1.4 Models (`run_hierarchical.py`) — 7 LightGBM fits, CPU
All fits share: `LGBMClassifier(objective=multiclass, n_estimators=800, learning_rate=0.07, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=30, n_jobs=8)`, with **inverse-frequency sample weights** (`1/(count+1)`, normalised to mean 1, computed after the cap on the 4x-duplicated rows). No early stopping, no validation set.
1. **Family model (8-way)**: families `[0]`, `[1..5]` jogging, `[6..10]` stretching, `[11,12]` push-ups, `[13,14]` sit-ups, `[16,17]` lunges, `[15]` burpees, `[18]` bench-dips.
2. **Variant models** for every family with >= 2 classes and >= 100 rows: jogging (5-way), stretching (5-way), push-ups (2), sit-ups (2), lunges (2). `p(c) = p(family) * p(c | family)`; singleton families get `p(family)` directly.
3. **Flat 19-class "champion"** LightGBM (same params, seed+99).
4. **Blend**: `probs = 0.7 * hierarchical + 0.3 * flat` (the CLI flag is misleadingly named `--video-weight`; it is the flat-model weight).
5. **Null bias**: `probs[:,0] *= exp(0.75) = 2.117`, renormalise, argmax. Written via `inference.write_submission` in test-meta row order (== id order).

### 1.5 Validation scheme and CV numbers
- **None on the executed path.** No CV, no OOF, no held-out subjects, fixed 800 trees. The grouped-CV code (`train.grouped_split`: 4 random held-out subjects, one fold, seed 42+fold; OOF per-class logit-offset coordinate ascent in `tuning.py`) exists only for the neural models and is never called by `run_hierarchical`.
- The author's own conclusions, from docstrings: `train_ensemble.py`: per-class offsets tuned on OOF "frequently overfit the held-out subjects and hurt the real test set"; `run_recording_select.py` title: "explains 'more data hurts' + CV-uselessness". I.e. **the public LB was his validation set.**
- LB numbers recorded in the code: reference (Nomannic port) **0.61278**; flat LightGBM champion **0.657**; adding more/uncapped training data **0.657 -> 0.586**; video-only transformer **~0.45**; submitted hierarchical **0.670** (kernel title). Expected-but-unmeasured: sensor specialists "+0.01 to +0.06", video-only "0.75-0.85 if video is the key".

### 1.6 Null-class handling
Null is an ordinary class 0: capped at 600 windows/subject, down-weighted by inverse frequency during training, then re-inflated by x2.117 at inference (value inherited from Nomannic's notebook, never re-tuned). There is **no per-subject null-rate matching** and no bout/boundary logic.

### 1.7 Post-processing and test-structure exploitation
- **Executed path: none.** Every test window is classified independently; no smoothing, no propagation, no subject-level constraints.
- Implemented but unused:
  - `temporal_smooth.py`: majority vote / probability average over +/-k neighbours after sorting each subject's windows by `id` — it assumes within-subject id order is temporal (unverified; the context says ids are shuffled across subjects, within-subject order unknown).
  - `run_propagate.py`: cosine similarity on video mean+std embeddings; windows with sim > 0.98 (`--dup-threshold 0.02`) from **different** sensors are forced to one label ("~243 cross-sensor duplicate moments").
  - `diagnose_duplicates.py` / `diagnose_groups.py`: hypothesis that the test set is "~3050 temporal moments x 4 sensors" and that the full 15x768 video can be used as a join key to recover 4-sensor groups.
- **That hypothesis is wrong**: per-subject window counts equal the summed session durations in seconds (sbj_23: 3128), so each of the 12,234 windows is a distinct second with ONE randomly chosen limb; 3048-3068 per sensor is simply 12,234/4. The "243 duplicates" are near-identical video from adjacent seconds of static activities, not the same instant seen by 4 sensors. Nothing in the submitted path depends on this, but it shows the author never found the real structure (one contiguous bout per activity per subject).

### 1.8 Why it reaches 0.670
1. **Classic WEAR-winner recipe**: rich per-limb time + frequency features (dominant frequency, band powers, entropy, zero-crossings, jerk, half-window trends) into gradient boosting. These features transfer across subjects far better than a raw-signal CNN trained for ~6 epochs on 22 subjects (his neural variants topped out around the 0.61 reference).
2. **Exact limb-matched training**: the 4-views-per-window construction means the test's single random limb always has a same-limb analogue in training, and the one-hot lets the trees specialise per limb without splitting the data 4 ways.
3. **Video mean/std adds the missing modality**: a single leg accelerometer cannot separate shoulder vs triceps stretching or push-ups vs bench-dips; the pooled VideoMAE vector encodes posture and gross motion and resolves most between-family confusions.
4. **Hierarchy + flat blend**: spends model capacity on the within-family confusions that dominate macro-F1 (5 jogging variants, 5 stretching variants, simple vs complex). Measured gain 0.657 -> 0.670.
5. **Near-balanced training** (600 cap + inverse-frequency weights) is the right objective for macro-F1; the exp(0.75) null bias restores null precision.
6. **Segment-aligned windows** give clean labels (no boundary-straddling windows).

### 1.9 Weakest links (ranked by likely gain if fixed)
1. **Video temporal structure is discarded** (15x768 -> mean/std -> fixed random 64-d projection). The 0.84-0.88 LB entries almost certainly model the video (posture + motion phase) far more thoroughly. His video-only transformer "0.45" is not the video ceiling: it was trained on <= 300 windows/class/subject for 5 epochs with no regularisation tuning.
2. **Per-window independence.** No use of the known test structure (each test subject performs each of the 18 activities exactly once as one contiguous bout, null in between). A per-subject bout decoder or even video-space kNN smoothing (our previous session gained +0.05-0.07 from that alone) is absent.
3. **No validation at all**, so 800 trees @ lr 0.07 / 63 leaves / null bias 0.75 / blend 0.3 are untuned; with no early stopping the model memorises subject idiosyncrasies. "More data hurts (0.657 -> 0.586)" is the signature of subject over-fit / covariate shift, not of harmful data.
4. **Null rate not matched per subject**; null ~45% of test (our estimate). Null errors leak into all 18 activity F1s at bout edges.
5. **NaN-sensor rows become zero-signal rows with real labels** on the boosting path (`nan_to_num` without the `valid` mask) — training label noise for subjects with dropped sensors.
6. **Single model per stage, single seed**; no seed bagging (`run_boost --ensemble` exists but `run_hierarchical` ignores it).
7. **Orientation-invariant features implemented but disabled**; sensor specialists, domain weighting, recording selection all implemented but not submitted (they evidently did not beat 0.670, magnitude unknown).

### 1.10 Directly reusable for us
- `features.py` (`extract_acc_features_vect`) — the 100-d per-limb block, vectorised; a drop-in upgrade over our 105-feature block (adds spectral entropy/centroid/rolloff/flux, band powers, half-window trends, magnitude stream).
- The `FAMILIES` hierarchy + 0.7/0.3 blend — cheap +0.01.
- The 600/(subject,class) cap + inverse-frequency weights + `exp(0.75)` null bias as a known-good starting operating point.

---

## 2. udaken10 / "Base_line_liner_model" (public LB unknown; no outputs, no score in title)

Japanese-language EDA + linear baseline; 17 cells (9 code), zero saved outputs.

### 2.1 Data and windowing (`build_train_dataset`)
- `max_files=3` on a sorted glob -> uses only **`sbj_0.csv`, `sbj_0_2.csv`, `sbj_1.csv`** (2 subjects, 3 recordings). `step_seconds=2` -> **one 1-s window every 2 s, aligned to even seconds** (no stride-25 overlap).
- Window label = **label of the middle sample** (`win_labels[25]`); windows may straddle a boundary. NaN label -> `str(nan)='nan'` -> `LABEL_MAP.get(..., 0)` -> null. Label map identical to akhyar's (null=0 ... bench-dips=18).
- Video for the train window = **all 30 frames** `sec*30 .. sec*30+30`, mean-pooled; test uses the central 15, also mean-pooled (roughly equivalent after averaging, but train sees +/-0.25 s more context).
- Each window -> 4 rows (one per limb) with a 4-d one-hot; `n_seconds = min(len(csv)//50, len(npy)//30)`.
- **No NaN handling anywhere**: `np.mean/std/min/max` propagate NaN; `StandardScaler` tolerates NaN but `LogisticRegression.fit` raises `ValueError: Input contains NaN`. Whether the kernel ran depends on those three recordings being NaN-free.

### 2.2 Features (784-d) and model
- 12 inertial (mean, std, min, max per axis) + 768 mean-pooled VideoMAE + 4 one-hot.
- `StandardScaler` -> `LogisticRegression(C=1.0, max_iter=500, random_state=42)` (lbfgs, multinomial). No class weights, no null handling, no post-processing.
- Only metric computed is **train** macro-F1 (`f1_score(y_train, model.predict(X_train_scaled))`); no CV, no held-out subjects. Prints predicted class distribution on test. Output `/kaggle/working/submission.csv`.

### 2.3 What it is good for
It is essentially a **linear probe on the 768-d VideoMAE mean vector** (the 12 IMU stats are a rounding error next to 768 standardised video dims), trained on two subjects. Alone it is weak, but it is maximally decorrelated from IMU-heavy boosters, which is why our worklog got +0.006 by blending it at weight 0.25 (sub d6 0.69065 -> d7 0.69707). Its LB is unknown.

---

## 3. Side-by-side

| | akhyar2612 (0.670) | udaken10 (unknown) |
|---|---|---|
| Train data | all 24 recordings, stride 25, in-segment windows, 600 cap/(sbj,class) | 3 recordings, one window per 2 s, middle-sample label |
| Inertial features | 100/limb (time+freq+trend on x,y,z,|a|) | 12/limb (mean,std,min,max) |
| Video | mean+std of central 15 frames -> random 64-d | mean of all 30 frames, 768-d |
| Limb handling | 4 views/window + one-hot | 4 views/window + one-hot |
| Model | 7x LightGBM (8-way family, 5 variant, flat) blend 0.7/0.3 | StandardScaler + LogisticRegression |
| Class balance | inverse-freq weights + cap | none |
| Null | class 0, x exp(0.75) at inference | class 0, nothing |
| Validation | none (LB only) | none (train F1 only) |
| Post-processing | none | none |
| Test structure used | none | none |
| Saved outputs | none | none |

---

## 4. Implications for a 0.85+ recipe
- Both kernels, and our own 0.715 lineage, treat windows independently and pool video to a vector. The gap to 0.88 is not feature engineering on the accelerometer; it is (a) modelling the 15x768 video sequence properly and (b) decoding each test subject's session as one contiguous bout per activity.
- akhyar's own evidence supports this: his best IMU+pooled-video model saturates at 0.657-0.670 across many variants (specialists, domain weights, recording selection, ensembles), i.e. the pooled-feature family is exhausted.
- Immediate cheap wins for our pipeline: adopt `features.py`'s 100-d block; add the family hierarchy; keep the 600 cap + inverse-frequency weights; then put the effort into a video sequence model and per-subject bout decoding rather than more IMU features.
