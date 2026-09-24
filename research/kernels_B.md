# Kernel analysis B: honghanhhh / nomannic19 / binasalama

Competition: `3rd-wear-dataset-challenge-hasca-2026` (macro F1, 19 classes). Written 2026-09-22.

Sources (local pulls via `kaggle kernels pull`, **all outputs stripped** -- 0 of 8 and 0 of 42 cells carry an `outputs` array, the .py script has none by construction, so NO printed metrics / CV numbers exist in any of the three files):

1. `E:\Claude code\wear\kernels\honghanhhh__wear-hasca\wear-hasca.ipynb` (8 code cells, 0 markdown) + `kernel-metadata.json` (CPU, no internet, `id_no` 134633536)
2. `E:\Claude code\wear\kernels\nomannic19__ts-emb-3wdc-temporal-fusion-ensemble\ts-emb-3wdc-temporal-fusion-ensemble.ipynb` (42 cells, 14 markdown) + `kernel-metadata.json` (T4 GPU, internet on, `id_no` 131619823, keywords gpu / time series analysis)
3. `E:\Claude code\wear\kernels\binasalama__wear-hasca-baseline\wear-hasca-baseline.py` (48-line script) + `kernel-metadata.json` (CPU, `id_no` 132257873)

LB scores: none of the three `kernel-metadata.json` files carry a score (the Kaggle metadata schema has no score field). Kaggle notebook pages are JS-rendered and returned only the `<title>` to WebFetch, so scores could not be scraped. The only score hints are inside the code itself:
- honghanhhh, cell 2 banner: `"If you do not see this banner in the logs, the old 0.639 notebook is still running."` -> the author's previous version scored **0.639** public LB; cell 7 prints `"0.80 needs the join to recover ~3000 groups of 4"` -> the author's target for this V8 was 0.80 and it depends on a test-structure hypothesis that is false (see below).
- binasalama, line 28 comment: `"simple static formula exactly as original best 0.03953"` -> **0.03953** LB (a no-learning constant-ish formula).
- nomannic19: no number anywhere; markdown states only dataset facts (null = 39.7 % of the train timeline, 24 recordings / 22 participants, 4 unseen test participants).

---

## 1. honghanhhh / "WEAR@HASCA" -- `V8_4IMU_VIDEO_JOIN_VITERBI`

### Idea in one line
Train a **4-limb LightGBM** on hand-crafted IMU features with **sensor-dropout** views, then at test time try to **re-assemble the 4 limbs of the same second by joining test windows through their VideoMAE fingerprint**, average probabilities inside the reconstructed "moment", and run a **Viterbi** smoother per subject ordered by `id`. **Video is deliberately NOT a classification feature** (`video_fingerprint` docstring: "Join key, not a class feature"; cell 6 title: "video is NOT a class feature").

### Models
- `lgb.LGBMClassifier(objective="multiclass", num_class=19, n_estimators=700, learning_rate=0.07, num_leaves=63, subsample=0.8, colsample_bytree=0.75/0.85, reg_lambda=1.0, min_child_samples=30)`, `N_SEEDS=2` boosters (seeds 42, 43), probabilities averaged.
- Class-balanced `sample_weight = 1/(count+1)`, normalised to mean 1 (`train_lgbm`).

### Inertial consumption
- Train windows: `WINDOW_SAMPLES=50`, `STRIDE=25` (0.5 s hop), **label-pure windows only** (`segment_starts` walks contiguous label segments, skips segments < 50 samples, aligns starts to multiples of the stride), NaN label -> class 0 (`label_ids`).
- Cap `MAX_PER_SUBJECT_CLASS=800` windows per (sbj_id, class) via random subsample (`subsample_subject_class`). With 22 subjects x 19 classes that is at most ~334k windows, in practice bounded by the shorter classes.
- Features per sensor, `imu_100`: for each of x, y, z and |a| -> `axis_block` = 13 time stats (mean, std, rms, min, max, range, median, skew, kurtosis, zero-cross rate, mean-cross rate, lag-1 autocorr, jerk std) + 8 spectral (dominant freq, centroid, spectral entropy, <2 Hz / 2-8 Hz / >=8 Hz band ratios, 95 % rolloff, flux) + 4 sub-window (half-window mean delta, std delta, energy ratio, linear slope) = 25 x 4 = **100 per limb**.
- `features_4`: 4 x 100 (missing limb -> zeros) + 6 pairwise magnitude correlations (`mag_corr_4`, zeroed when either limb absent) + 4-bit presence mask = **410 features**.
- Sensor dropout (cell 5): the full 4-limb view + `N_DROPOUT_COPIES=2` copies where each window keeps k in {1,2,3} limbs with p = [0.30, 0.35, 0.35]. So train matrix = 3 x n_windows rows. Note only ~30 % x 2/3 = 20 % of rows are single-limb, whereas **100 % of real test windows are single-limb** unless the join works.

### Video consumption
- Test video transposed `(N,768,15) -> (N,15,768)`, then `video_fingerprint` = L2-normalised concat of frame-mean and frame-std (1536-d). Used **only as a join key**; the 15-frame temporal structure is only pooled. Train video .npy files are inventoried (cell 4 prints shapes) but **never loaded for features**. `VIDEO_FRAME_OFFSET=8` / `VIDEO_WINDOW=15` are defined but unused.

### Test-structure exploitation (the core bet) -- cell 7
Three candidate groupings of test windows into "moments" (same subject, distinct limbs, <= 4 windows):
1. exact hash of the full 15x768 float32 block (`pd.util.hash_pandas_object`) -> `groups_from_keys` (same sbj + same key, one window per limb);
2. hash of the fingerprint rounded to 5 decimals;
3. greedy 1-1 cosine matching across every limb pair within a subject with union-find, thresholds `(0.9995, 0.999, 0.997, 0.995, 0.99, 0.98)`, never merging two windows of the same limb and capping a group at 4.
The candidate with the best key `(exact4 groups, windows covered by >=4-sensor groups, 3-sensor groups, -oversize)` is selected; a WARNING is printed if fewer than 200 exact 4-limb moments were found.

**Why this cannot work as intended on this test set:** per the verified facts, window counts per subject equal the summed session seconds (e.g. sbj 23 = 3128 s = 3128 windows) and each window is ONE randomly chosen limb, so every second appears exactly once -- there are no other-limb duplicates of the same second to join. Methods 1-2 will yield 0 groups >1; method 3 at thr 0.98 can only pair *adjacent/nearby seconds* whose VideoMAE clips happen to be near-identical (nomannic19's EDA computes `adjacent_cosine` precisely because consecutive VideoMAE frames are highly correlated). That is an approximate neighbour-limb reconstruction (activities are contiguous bouts, so a neighbouring second usually has the same label) but the IMU of a neighbouring second is not the same second's IMU, and the model was trained on truly synchronous limbs.

### Post-processing (cell 8)
- `NULL_BIAS=0.35`: `probs[:,0] *= exp(0.35)` (x1.42) then renormalise.
- Probability averaging over all members of a moment.
- **Viterbi per subject** (`STAY_PROB=0.93`, uniform off-diagonal `(1-0.93)/18`), emissions = moment-mean probabilities, **timeline ordered by the minimum `id` in the group**. This assumes `id` order is temporal within a subject -- flagged UNKNOWN in our facts (ids are interleaved across subjects: id 0 = sbj 24, id 1 = sbj 23, ...). If within-subject order is not temporal, stay=0.93 smoothing over a shuffled sequence would damage predictions; the author hedges by also writing `submission_nosmooth.csv` (raw argmax).
- Submission written from `sample_submission.csv` with `target_feature` mapped by id; asserts range 0..18.

### Validation
**None.** No holdout, no leave-subject-out, no printed F1. The only feedback loop is the LB (0.639 previous version).

### Verdict
Nice IMU feature bank (the 100-d FAME block is a reusable component, richer than our previous stats) and a correct sensor-dropout idea, but it throws away the strongest modality (video) as a class feature, bets on a join that the test construction rules out, and applies Viterbi over an id order of unknown meaning. Expected LB: <= our 0.715 range, probably below the author's own 0.639 unless the greedy neighbour matching accidentally acts as label smoothing.

---

## 2. nomannic19 / "[TS/Emb] 3WDC | Temporal Fusion Ensemble" (most-voted)

### Idea in one line
Two small PyTorch fusion nets trained on **all 4 limbs of every window as 4 separate samples** (shared video clip, per-limb IMU, learned sensor-location embedding), then a **0.6 / 0.4 probability blend** with a **null bias of exp(0.75)**. Test windows are fed as single-limb samples with their `sensor_location` id.

### Data / windowing (cells 27-28)
- IMU cached per recording as `(T, 4, 3)` float32 `.npy` in `/kaggle/working/inertial_cache`.
- Label-pure windows: 50 samples, `WINDOW_STRIDE=25`, starts aligned to multiples of 25 inside each contiguous label segment; NaN label -> "null".
- `video_start = inertial_start * 3 // 5` (50 Hz -> 30 FPS) and the clip is `video[video_start+8 : video_start+23]` = **frames 8..22 of the 30-frame second = the central 15 frames**, which the markdown states matches the test format ("VideoMAE indices 8-22 provide the 15 valid features available in test data"). This is the exact train/test alignment recipe to reuse.
- Cap `WINDOWS_PER_SUBJECT_CLASS=150` per (sbj_id, class) -> at most 22 x 19 x 150 = 6,270 windows -> 25,080 limb-samples (the notebook prints `Temporal windows: ... | Sensor samples: ... x4`). This is a **tiny** training set (<3 % of the available label-pure windows) -- the notebook is a fast demo, not a tuned solution.
- `valid_sensors` mask (all-finite check per limb) excludes limb-samples with NaN IMU from the loss.
- Smoke mode: 2 windows per class, 1 epoch.

### Model A -- `PooledFusionModel` (cell 32)
- IMU branch: 16 hand stats per limb (mean/std/min/max of x,y,z + mean/std/min/max of |a|) -> Linear 16->64 + LayerNorm + GELU + Dropout 0.15.
- Video branch: **pooled** concat(frame-mean, frame-std) 1536 -> LayerNorm -> Linear 1536->192 -> GELU -> Dropout 0.2.
- Sensor embedding `nn.Embedding(4, 8)`.
- Classifier: concat (64+192+8=264) -> 128 -> 19.
- Trained 3 epochs (`POOLED_EPOCHS`), AdamW lr 2e-3, wd 1e-4, no scheduler.

### Model B -- `TemporalFusionModel` (cell 33) -- the part that uses the 15x768 sequence
- `InertialEncoder`: input (3+|a|=4 channels, 50 steps) -> 2 x `InceptionBlock` (1x1 bottleneck to 32 ch, parallel Conv1d kernels 5/11/21 + maxpool branch, GroupNorm(8), residual skip, Dropout 0.1; 128 output ch) -> concat(mean-pool, max-pool) 256 -> Linear 192 + LayerNorm + GELU + Dropout 0.2.
- `VideoEncoder`: per-frame LayerNorm(768) -> Linear 768->192 -> GELU, + learnable positional embedding `(1,15,192)`, **1-layer `nn.TransformerEncoder` (d=192, 4 heads, ff 384, dropout 0.15, pre-norm)** over the 15 frames, then concat(attention-pooled with a Tanh MLP scorer, mean-pooled) 384 -> Linear 192 + LayerNorm + GELU + Dropout 0.2.
- Fusion: sensor embedding (4,16); sigmoid gate over concat(imu 192, video 192, sensor 16 = 400) -> 192; `fused = g*imu + (1-g)*video`; classifier input concat(fused, imu*video, sensor) = 400 -> 192 -> 19 (Dropout 0.3).
- Augmentation in `train()` mode: per-limb amplitude scale U(0.9,1.1) + Gaussian noise sd 0.01 on raw IMU; **modality dropout 0.1** on the IMU embedding (per limb) and the video embedding (per window), inverse-scaled.
- Trained 4 epochs, AdamW lr 8e-4, wd 1e-3, CosineAnnealingLR with `T_max=10` (so only 4 of 10 cosine steps are taken), fp16 autocast + GradScaler, `nn.DataParallel` if >1 GPU. Parameter count is printed (not preserved).
- Loss for both models: `CrossEntropyLoss(label_smoothing=0.05)`, target repeated 4x (`repeat_interleave(sensors)`), masked by `valid`.

### Inference / post-processing (cells 39-41)
- Test IMU `(50,3)` -> `.T[None]` = `(1,3,50)`; video `(768,15).T` -> `(15,768)`; `sensor_ids` from `test_meta.sensor_location` passed explicitly.
- `combined = 0.60 * softmax(temporal) + 0.40 * softmax(pooled)`; `combined[:,0] *= exp(0.75)` (x2.12 on null) ; argmax. No smoothing, no per-subject logic, no test-structure exploitation (ids/subjects are ignored beyond the sensor id).

### Validation
**None.** `fit_model` records only the training loss per epoch (`train_loss`); no holdout, no leave-one-subject-out, no macro F1 is ever computed. The markdown warns that overlapping windows "must never be randomly divided between training and validation", but then no validation is implemented at all.

### EDA facts stated in markdown (cell 23) that we can bank
- 24 recordings from 22 participants; null = 39.7 % of the training timeline; the 18 activities are "relatively balanced".
- Inertial 50 Hz and video 30 FPS are aligned exactly; `duration_gap_s` per recording is computed (values not preserved).
- Train vs test distributions of pooled-norm / temporal-std / adjacent-cosine of VideoMAE and of mean-magnitude / dynamic-RMS / jerk-RMS per sensor location are "similar for both modalities"; per-sensor normalisation is recommended.
- "Consecutive VideoMAE features are highly correlated" -> the author argues for pooling or lightweight temporal attention rather than a big transformer.

### Verdict
The most useful of the three as a **template**: exact 15-frame alignment recipe, 4-limbs-as-4-samples training with a sensor embedding (this is the right way to make single-limb test windows in-distribution), modality dropout, gated fusion, Inception-1D IMU encoder + 1-layer temporal transformer over the VideoMAE frames. But as shipped it is badly under-trained (6.3k windows, 3-4 epochs, no CV) and its null bias (x2.12) is a blind guess. Nothing in it explains 0.88; it needs (a) all label-pure windows (100k+), (b) 20-40 epochs with LOSO validation, (c) per-subject null-rate calibration.

---

## 3. binasalama / "WEAR HASCA Baseline" (script)

- No learning at all. `inertial_feat = ||inertial||` over the whole (50,3) window; `videomae_feat = mean over all 768x15`; `combined = 0.6*(inertial/max*18) + 0.4*(video/max*18)`; `preds = clip(round(combined), 4, 18)`. Class ids 0-3 can never be predicted.
- Row order of `sample_submission.csv` is assumed to be the row order of the .npy arrays ("no shuffle, row order preserved"), which is true only because sample_submission ids are 0..N-1 in order.
- Fallback: constant class 7 if anything fails. A stray `agent(obs, config)` stub returning `{"action": 0}` (copied from a simulation competition).
- Self-reported LB in the comment: **0.03953**. Zero value for us other than confirming that macro F1 ~0.04 is the floor.

---

## Cross-kernel takeaways for the rank-1 push

1. **Train/test video alignment is `frames[start*3//5 + 8 : +23]`** (nomannic19) -- the central 15 of the 30 frames of the inertial second. Our previous pipeline pooled the 15 frames; both kernels also pool (honghanhhh) or use only a 1-layer transformer (nomannic19). Nobody here trains a proper sequence model on the 15x768 block with enough data.
2. **Single-limb test windows should be simulated in training** either as 4 separate limb-samples with a sensor embedding (nomannic19) or as sensor-dropout views of a 4-limb feature vector (honghanhhh). Our previous limb-matched single-sensor LightGBM is a third way; the sensor-embedding net is the cleanest.
3. **Both real kernels skip validation entirely** and hand-tune a null multiplier (exp(0.35) vs exp(0.75)). Since every test subject performs each of the 18 activities exactly once as one contiguous bout, the true per-subject null fraction is (session seconds - 18 bout durations)/seconds; a LOSO-calibrated null threshold or the null-rate matching we already have is strictly better than a blind bias.
4. **honghanhhh's 100-d FAME IMU block** (time + spectral + sub-window stats on x,y,z,|a|) is a drop-in upgrade for our hand-crafted IMU features.
5. **Test-structure bets to avoid**: joining limbs across test windows by VideoMAE identity (impossible: one window per second) and Viterbi along `id` order (order unknown). The safe structural lever remains video-space kNN / clustering within a subject (contiguous bouts -> near-duplicate VideoMAE clips share a label), which honghanhhh's greedy cosine matching approximates by accident.
6. None of the three files contains any output cell, CV number or LB number beyond the two comment-embedded scores (0.639 prior honghanhhh version; 0.03953 binasalama). Vote counts / current LB of these kernels are not recoverable offline (Kaggle pages are JS-rendered; WebFetch returned only titles).
