# WEAR challenges: literature and results sweep (2024 -> 2026)

Compiled 2026-09-23 for the 3rd WEAR Dataset Challenge @HASCA 2026 (Kaggle slug `3rd-wear-dataset-challenge-hasca-2026`, macro F1 over 19 classes, closes in ~20 days; public LB top = 0.88425 thisray, 0.88126 An Dao, 0.87430 Benjamin Beaujard, 0.87059 habakan, 0.86687 anhnam_xtanh, 0.86584 T Hansda, 0.86132 Vedanam Coders, 0.85796 Gleb Shanshin, 0.84551 KMET, 0.84061 Mateo Allmer; 131 teams / 464 entrants / 2,693 submissions as of today, from `kaggle competitions leaderboard -s` and the Kaggle overview page).

All PDF text was extracted with pypdf into `C:\Users\Koushik\AppData\Local\Temp\claude\E--Claude-code\4892a651-709e-47dc-8a3d-bbb791e079fe\scratchpad\` (`webfetch-*.txt`, `wear_supp.txt`, `levienbang_*.txt`, `0-670.txt`, `ts-emb-3wdc-temporal-fusion-ensemble.txt`, `wear-hasca.txt`).

---

## 0. TL;DR (what the literature says a rank 1-3 recipe must contain)

1. **Feature-rich GBDT on the single 1 s IMU window is the proven backbone** (2024 winner: CatBoost on 14 tsflex features x multi-window; 2025 winner FAME + every 2025 report: frequency-domain + statistical + SMV/angle features into LightGBM/XGB/HGB). Single-sensor 1 s inertial-only ceilings on unseen subjects: 0.57-0.62 (2025 private LB), 0.62 arm / 0.57 leg LOSO (rTsfNet), 0.67-0.70 val (2026 public notebooks).
2. **Video is the 2026 lever.** The WEAR paper's own 1 s test-set numbers: inertial-only (all 4 sensors + 10 s majority vote) F1 86.20, camera-only F1 72.49, early fusion (TriDet, I+C) F1 86.98 / mAP 93.12, and an *oracle* late fusion of I, C and I+C reaches F1 93.38 in LOSO -> the modalities are complementary and the dataset is "far from saturated". A 2026 participant repo (yuma-iwamoto/WEAR2026-UEC-dx2) reaches 0.7527 val macro F1 on held-out sbj 18-21 with a 6-model beam-weighted ensemble in which the two best-weighted members are Inertial-LightGBM (w=0.27) and Inertial-CNN+Video-MLP (w=0.22), and the Video-MLP alone gets w=0.16.
3. **Temporal context is the biggest single gain in every prior WEAR result** (2024 winner: multi-window features over {1,2,4,8,16,32} s and Gaussian smoothing over +-5 s; WEAR paper: 10 s majority vote lifts DeepConvLSTM 1 s-window F1 from 70.36 (raw) to 75.44 and mAP from 9.26 to 59.14). The 2025 challenge *removed* context (random shuffled windows) and scores collapsed to 0.60. The 2026 test tiles every second of every session with stride 1 s (12,234 windows = summed session seconds), and the 15 VideoMAE frame features overlap across neighbouring seconds, so **recovering the per-subject timeline (video-feature continuity / kNN chaining) and then smoothing (majority vote / Viterbi with stay prob ~0.93) is the most likely explanation for the 0.84-0.88 cluster with 4-6 submissions**. Public notebooks already attempt this (honghanhhh V8: "Viterbi / subject stay=0.93"; akhyar2612: "Sorting each subject's windows by id recovers its 1-second time axis; adjacent windows are the same activity ~90%+ of the time").
4. **Null handling**: undersample null to ~4x the mean activity-class count (HARMA 2025: +1.2 pt), or per-class logit offsets / null threshold tuned on OOF (HARMA +1.1 pt; combining both gave nothing extra). Team HARMA's 2026 code applies "3x median null undersampling". The 2024 winner's rule "every activity is present >= 50 s per participant recording" is protocol-exploiting and still true (each activity done for ~90 s in 3 sets of ~30 s).
5. **Orientation/limb augmentation**: left-right swapping (+1.3 pt) and upper-lower pairing (+1.9 pt) in 2024; axis inversion for arm x / leg y and 180-degree rotation about x (3KA 2025); channel-wise random sign flipping (FAME 2025 winner, "main driver of generalization"); jitter/scaling/+-15-degree rotation/channel dropout (PatchTST 2025, +1.6 pt).

---

## 1. WEAR dataset paper (arXiv 2304.05088 v4 / IMWUT 8(4) art. 175, DOI 10.1145/3699776) and supplementary

Sources: https://arxiv.org/abs/2304.05088 (PDF text extracted), supplementary `https://raw.githubusercontent.com/mariusbock/wear/main/supplementary_material.pdf` (27 pages, extracted to `wear_supp.txt`), project page https://mariusbock.github.io/wear/, code https://github.com/mariusbock/wear, dataset https://bit.ly/wear_dataset (annotations 14 MB, processed I3D/inertial 44 GB, raw 164 GB).

### 1.1 Recording protocol (answers the "same order? bout durations? null between bouts?" questions)

- 22 participants (13 m / 9 f), 18 activities, 11 outdoor locations; "first 18 participants ... over 5 months (October to February), totalling more than 15 hours"; then "an additional 6 participants were recorded, totalling around 4 hours" = 4 new subjects + re-recordings of sbj_0 and sbj_14 in August (these are `sbj_0_2.csv`, `sbj_14_2.csv` in the 2026 train set; the paper's own test split calls them sbj_18/19 with asterisks in Table 2).
- Sensors: 4 Bangle.js v1 watches, 50 Hz, +-8 g, "placed by the researchers in a fixed orientation on the left and right wrists and ankles"; true rate ~48 +-1 Hz, resampled to 50 Hz by linear interpolation between sync jumps. Camera: GoPro Hero 8 (train) / Hero 11 (4 new subjects), head-mounted, tilted ~45 degrees down, 1080p 60 FPS resampled to 60 FPS (challenge features are at 30 FPS).
- **Order is NOT fixed.** Main paper: "Participants were suggested to follow a two-session setup, i.e. 9 activities per session. Nevertheless, it was allowed to differ from this setup and split the 18 activities across as many (or as few) sessions". 2024 winner's paper confirms: "Although each participant performed the same set of 18 workout activities, they had the freedom to determine the exact activity sequence and how each sequence was performed". The recording plan (supp. Section D, Figs 9-13) lists a *suggested* grouping: **1st session (ca. 30 min)**: running sidesteps, butt-kicks, stretching shoulders, hamstrings, lumbar rotation, burpees, lunges (normal + complex), bench dips = 9; **2nd session (ca. 30 min)**: jogging, jogging rotating arms, skipping, stretching triceps, lunging, push-ups (normal + complex), sit-ups (normal + complex) = 9. The test subjects' 9/8/1 and 9/9 session splits (participant_meta_data.txt) match this two-session plan.
- **Bout structure: each activity is ~90 s but NOT one contiguous 90 s bout.** Recording plan: "Each activity 3 sets a repetitions at will (ca. 30 sec); short break after each set (ca. 30 sec)". Main paper: "roughly 90 seconds ... it was not required to perform activities for 90 seconds straight and participants could include breaks as needed". Supplementary nutrition label: **751 action segments** over 1137 min for 22 participants -> 751 / (22 x 18) ~ 1.9 labelled segments per activity per subject, i.e. set breaks are often (not always) annotated as null. Null = **37 %** of the dataset (Fig. 2, supp.); every activity 3-4 %. (The 2026 Kaggle train CSVs give 39.7 % null per the nomannic19 notebook.)
- Annotation: one expert annotator in Final Cut Pro from the third-person tripod video; gaps between subtitles -> NULL.
- Per-session table (supp. Table 2): most sessions 9 activities, 16-36 min; e.g. sbj_0 did 3 sessions (7/6/7 activities), sbj_3 did 4 (10/6/2/2), sbj_10 3 x 9, sbj_2 9/9/1. Locations 1-11 in train; the 2026 test uses locations 12, 13, 14, 15 (new) and 3 (seen: "small square with concrete surface").

### 1.2 Baselines (Table 2, LOSO on the first 18 participants, 3 seeds, all 4 sensors)

| Model | clip | P | R | F1 | avg mAP |
|---|---|---|---|---|---|
| Shallow DeepConvLSTM (inertial) | 0.5 s / 1 s / 2 s | 74.81 / 77.71 / 78.00 | 74.93 / 77.41 / 77.19 | 72.26 / **75.44** / 75.43 | 55.99 / 59.14 / 61.25 |
| Attend-and-Discriminate (inertial) | 0.5 / 1 / 2 s | 76.54 / 79.97 / 80.96 | 73.04 / 75.92 / 78.42 | 72.10 / 74.67 / **77.10** | 52.15 / 56.44 / 59.04 |
| ActionFormer (inertial, vectorised 600-d) | 1 s | 65.88 | 78.44 | 68.40 | 75.01 |
| TriDet (inertial) | 1 s | 68.18 | 78.58 | 70.36 | 76.86 |
| ActionFormer (camera, I3D 2048-d) | 1 s | 65.82 | 75.34 | 66.40 | 79.32 |
| TriDet (camera) | 1 s | 67.42 | 75.21 | 66.88 | **81.30** |
| ActionFormer (I + C early fusion, 2648-d) | 1 s | 72.87 | 83.24 | **75.26** | 81.03 |
| TriDet (I + C) | 1 s | 73.39 | 82.00 | 75.09 | **82.26** |
| Oracle late fusion O-LF(I, C) | 1 s | 91.26 | 92.72 | 90.68 | 73.52 |
| Oracle O-LF(I, C, I+C) | 1 s | 93.99 | 94.89 | **93.38** | 83.13 |

Post-processing baked into those numbers: inertial models smoothed with a **10 s majority-vote filter**; TAL models use score threshold 0.1 to create NULL. Supplementary Table 5 (Shallow DeepConvLSTM, 1 s window, 50 % overlap): no filter F1 70.36 / mAP 9.26; 5 s 75.04 / 51.08; **10 s 75.44 / 59.14**; 15 s 75.31; 20 s 75.05; 25 s 74.66 -> smoothing is worth ~+5 F1 and 10 s is the sweet spot. Feature note: I3D clips of 0.5/1/2 s with 50 % overlap; inertial windows are simply vectorised (window length x 12 axes).

Per-class findings: video-only models "struggle to differentiate between different running styles, activities outside the field of view (e.g., triceps stretches), as well as normal and complex sit-ups"; inertial models are "particularly reliable during activities where limb orientation is the main discriminator (e.g., stretches)" but confuse activities with NULL; vision has "a larger NULL-class accuracy". Fusion removes the jogging confusions and reduces NULL confusion. Sensor ablation (supp. Table 9): right wrist only DeepConvLSTM 1 s F1 61.97 (vs 75.44 with 4 sensors); right wrist + right ankle 73.54.

### 1.3 Test set (4 unseen + 2 re-recorded participants, spring/summer, 1 s window, 50 % overlap; supp. Table 4 / Fig. 6)

Inertial DeepConvLSTM 1 s **F1 86.20** (mAP 71.09); A-and-D 1 s 83.97; camera TriDet 1 s F1 72.49 / mAP 85.48; **I + C TriDet 1 s F1 86.98 / mAP 93.12**, ActionFormer I+C 84.26 / 92.98. Repeated sessions (sbj_0, sbj_14 in August): improvement only for sbj_0 (knew 5 of 18 exercises beforehand); differences "within the expected standard deviation across participants (between 15 % to 20 %)".

### 1.4 Mapping to the 2026 test format

- 2026 gives **one random limb** (3 axes) per 1 s window instead of 12 axes, no context beyond the window, and the **central 15 of 30** VideoMAEv2-Base 768-d frame features (each frame feature already sees frames i-8..i+7, i.e. the 15 kept frames cover ~0.75 s +- 0.27 s of video around the window centre). Kaggle data page: "Windows contain 15 feature vectors instead of the 30 to avoid frames near the boundaries of a window to leak information about past/future context". Train `.npy` = 30 FPS per-frame features, index mapping `video_idx = inertial_idx * 3 // 5`, and public notebooks take frames `[v+8, v+23)`.
- Consequence: the paper's inertial 86.20 assumed 4 sensors + 10 s smoothing; the single-sensor no-context analogue is the 2025 challenge (0.57-0.62). The video side is closer to the paper's camera numbers (TriDet camera 1 s F1 72.49 on the test set, with a much stronger backbone here: VideoMAEv2-Base vs I3D).
- Test = 12,234 windows = summed session seconds (sbj_22: 40:30+40:10+2:57 = 5017 s vs 5197 windows reported in context - 180 s difference, sbj_23: 3128 exact, sbj_24: 17:25+15:50 = 1995 exact, sbj_25: 16:05+15:49 = 1914 exact) -> stride-1 s tiling; ids are shuffled across subjects, in-subject order unknown (see open questions).

---

## 2. 1st WEAR challenge 2024 (inertial, 4 sensors, untrimmed streams, sample-wise macro F1)

**Winner "Signal Sleuths" (Jonas & Jeroen Van Der Donckt, Sofie Van Hoecke, Ghent-imec)**, arXiv 2408.03947, DOI 10.1145/3675094.3678453. https://arxiv.org/abs/2408.03947 (PDF text in `webfetch-1790099759024-cognjb.txt`).

- Train = 18 participants, test = 6 recordings (2 are re-recordings). Evaluation sample-wise macro F1.
- EDA: wearable orientation inconsistent (participant 2 arm watches inverted; participant 9 left leg inverted; participant 5 and test participant 18 change orientation between sessions; participant 10 has a missing left-arm segment imputed from the right arm; test participant 19's first-session legs nearly flat).
- **Features (N = 14 per window)**: time domain on raw axes and SMV: min, max, ptp, iqr, std, skew, kurtosis, Hjorth mobility & complexity, mean-crossing rate, differential entropy, Petrosian FD, Katz FD; PSD spectral entropy (not on 1 s). Libraries: tsflex, antropy, numpy/scipy.
- **Multi-window context**: for each prediction time step, past *and* future windows of {1, 2, 4, 8, 16, 32} s -> 6 x 2 windows -> 166 features per axis -> **1992 features** (166 x 3 axes x 4 wearables); stride 0.5 s.
- Model: **CatBoost**, 1,000 iterations, depth 5, border_count 32, auto_class_weight="Balanced", defaults otherwise. Whole pipeline < 1 h on a Ryzen 5 2600x.
- Augmentation: (a) rotation-invariant stats {mean,std}/{mean,std,skew}/{min,mid,max} over x,y,z (worse than raw by ~1 pt -> orientation info matters); (b) **LR-swapping**: augment with {no swap, upper swap, lower swap, both} and majority-vote the 4 views at test; (c) **UL-pairing**: feature vectors from one arm + one leg (4 pairs -> 4x rows, half the features 996), aggregate over pairs at test.
- Post-processing: 3-fold subject-grouped CV models majority-voted (+1 pt); **temporal smoothing with a Gaussian kernel, sigma = 6 over a 10-10 step receptive field (5 s each side at 0.5 s stride)** (+0.3 to +0.9 pt); **rule-based boosting**: "each of the 18 workout activities were present for at least 50 seconds for every participant recording" -> force under-predicted classes into regions where they have mass (not applicable in practice).
- Results (grouped 3-fold, F1 / F1 with smoothing): SMV 0.8055/0.8146; rot_inv_stat2 0.8806/0.8882; rot_inv_sort 0.8815/0.8891; rot_inv_stat3 0.8837/0.8904; raw 0.8953/**0.9001**; LR-stacking 0.9084/**0.9130**; UL-pairing 0.9154/**0.9187** (sample-wise expanded 0.9185). Baseline they beat: A-and-D 83.08 %.
- Confusions: "limited confusion among activity classes but substantial confusion with the null class. This is especially pronounced for stretching-based activities" (triceps 0.79-0.83 recall, lunging 0.78-0.83, shoulders 0.72-0.74, sit-ups 0.81-0.82; jogging variants 0.94-0.99); lunges vs complex lunges +5 pt from augmentation.

---

## 3. 2nd WEAR challenge 2025 (inertial only, random shuffled 1 s windows, ONE sensor per window, 4 unseen subjects)

Kaggle https://www.kaggle.com/competitions/2nd-wear-dataset-challenge - public LB (kaggle CLI): Whatever 0.60249, Schw1Eg0 of the White Forest 0.58750, HARMA 0.57773, Signal Sleuths 0.57657, Sumitou Atsuya 0.57485, 3KA 0.57441, ..., VedaNam Coders 0.53017. Private LB ("approximately 50 % of the test data") is not exposed by the API; the reservoir paper quotes its own private/test score 0.52888. Same data structure as 2026 minus video: HARMA's report lists the three specific difficulties "Lack of Temporal Context", "Availability of Only One Sensor Location at a Time" (location given), "Unseen Test Subjects".

### 3.1 Winner "Whatever": FAME (Calatrava-Nicolas, Ray, Fortes Rey, Lukowicz, Martinez Mozos; DFKI / Orebro), DOI 10.1145/3714394.3756194 (pp. 964-969)

https://doi.org/10.1145/3714394.3756194 (ACM full text blocked by Cloudflare for both WebFetch and the browser; abstract via Semantic Scholar API and https://biblio.ub.rptu.de/frontdoor/index/index/docId/18934), code https://github.com/FranciscoCalatrava/FAME-Feature-Augmented-Multi-View-Ensemble (single 372-byte Readme, no code pushed yet).
- "multi-view ensemble model considering each view as the set of symmetrically worn sensors (i.e. right/left arm)"; "an early encoder that shares weights across the views, followed by two view-specific branches (wrist and legs)"; "encourages similarity between symmetric sensor positions by adding a similarity component to the loss function"; "enriches inputs with frequency-domain and PCA features plus data augmentation".
- Key finding: "**frequency-domain features and channel-wise random sign-flipping data augmentation were the main drivers of generalization**, mirroring patterns seen on the challenge test set". Public LB 0.60249 (rank 1). "Fran (Team Whatever)" is also credited on the 2026 discussion board for finding the test video-feature bug, so this team is active in 2026 (team name on 2026 LB unknown).

### 3.2 Team HARMA (Link & Stuckenschmidt, Mannheim), DOI 10.1145/3714394.3756192, also https://d-nb.info/1386433594/34 (full text in `webfetch-1790099732326-8hacw2.txt`), code https://github.com/rilink/HARMA

- 1 s samples, each sensor location a separate training row (4x rows), **one LightGBM jointly across locations** (location as a feature); subject-based 70/15/15 split then 6-fold CV for hyperparameters; ensemble = soft vote of LightGBM + TabM + GPBoost.
- Features: per-axis time (mean, var, std, RMS, shape factor, max, min, median, IQR, SMA, skew, kurtosis, jerk mean/std, #peaks, zero crossings) + per-axis frequency (energy, dominant frequency, spectral entropy, signal entropy) + multi-axis (sums, Euclidean mean, signed/absolute magnitude, axis correlations/covariances, Welch spectral entropy of SMV, SMV min/max/ptp/IQR/std/skew/kurt, Hjorth mobility/complexity, mean-crossing rate, differential entropy, Petrosian/Katz FD, **tilt angle, pitch, roll**, double-integrated position/velocity). Full list: HARMA-supp.pdf (extracted).
- Null strategies (test macro F1, Table 1): baseline LightGBM 0.573; drop null + confidence threshold **0.540** (worse; "null class may in fact exhibit some regularity"); **adjusted undersampling 0.585** (best at mean-class-count + 20,000 null rows ~ 4x the other classes; full undersampling to the mean was worse than baseline); confidence thresholding with a separate null threshold h0 and a shared activity threshold 0.584; both combined 0.585 (no gain); ensemble under undersampling **0.594**.
- 2026 version of HARMA (repo README, updated 2026-07-08): OOF stacking of LightGBM-IMU (0.591), IMU + 64-d **temporal PCA of the flattened 15x768 VideoMAE window** (0.663, PCA fit transductively on train+test, per-subject residualisation), IMU + 19-class softmax from a **subject-adversarial DANN video encoder** (LSTM-128 over the 15 frames, 64-d latent, GRL lambda ramp 2/(1+e^-10p)-1, 20 epochs) (0.683); Ridge meta-learner 0.725 OOF / **0.732 LB**, max-confidence fusion over LightGBM/GPBoost/TabM meta-learners **0.737 LB**. LightGBM params: n_estimators 400, max_depth 15, lr 0.1, subsample 0.9, min_child_samples 20, reg 0.1; "3x median null-class undersampling". Windows require >= 40/50 samples of one label. No temporal modelling -> stuck at 0.737 (not in the current top 20).

### 3.3 Team 3KA (Phan, Le, Nguyen, Dao, Le; HCMUT), arXiv 2511.23173, DOI 10.1145/3714394.3756193

https://arxiv.org/abs/2511.23173, code https://github.com/Khanghcmut/WEAR-Challenge. Note the 2026 LB #2 "An Dao" (username andao94) and #5 "anhnam_xtanh" are plausibly the same Vietnamese group (Anh Van Dao is a 3KA co-author) - unverified.
- **Data-side fusion**: merge left and right sensors of each limb -> one arm model and one leg model; **augmentation**: invert x for arm-worn data and y for leg-worn data (simulates opposite limb), and a 180-degree rotation about x (upside-down watch).
- **450 features**: 135 raw-axis (45 per axis: 27 TSFEL statistical/temporal, 4 fractal/spectral - Petrosian, Katz + 2 spectral, 14 higher-order differential), 180 SMV-based (full SMV + 2-axis variants, 45 each), 135 angle-based (arctan2 of axis pairs, 45 each).
- Models: HistGradientBoosting (balanced class weights) + XGBoost (unweighted), soft-voting average. Group 5-fold by participant: XGB arm 60.25+-2.52 / leg 54.40+-3.49; HGBC 58.73 / 52.99; **voting 61.72 / 55.95 (avg 58.84)**. Null confusion: "the model often misclassifies activities with minimal leg movement (e.g., stretching or bench dips) as 'Null'". Arm: fractal/spectral features most important (~3x the leg F-score). No post-processing. Public LB 0.57441.

### 3.4 PatchTST ensembles (Chandankar & Burchard, Siegen), arXiv 2510.21282

https://arxiv.org/abs/2510.21282 (HTML version readable).
- 4 sensor-specific PatchTST encoders (patch L=5, P=10 patches, d=128, 4 layers, 8 heads, 0.74 M params each), 50 epochs AdamW lr 3e-4, cosine to 1e-6, label smoothing 0.10, dropout 0.1, stochastic depth 0.05, grad clip 1.0; per-window z-score (+0.53 pt over global z-score); training windows 50 samples, stride 25 (0.45-0.88 M windows).
- **Test-matched augmentation** ("tampered training set"): Gaussian jitter sigma ~ U(0.02, 0.04) g, scaling s ~ U(0.9, 1.2), rotation theta ~ U(-15, 15) degrees, channel dropout p = 0.20; ablation (val F1): none 45.36, +jitter 50.32, +rotation 51.12, +scaling 50.02, +dropout 48.15, all four **52.72**.
- Dual stream (clean + robust model per sensor, 8 probability vectors averaged), per-fold temperature scaling (ECE 9.4 % -> 3.6 %). 5 subject-exclusive folds: baseline 0.551 -> +aug 0.557 -> +per-window norm 0.562 -> +dual stream 0.569. **Hidden test (Table 4)**: DeepConvLSTM 0.4428, TinyHAR 0.4702, A-and-D 0.4718, PatchTST clean 0.5098, robust 0.5123, dual-stream **0.5172**. No temporal post-processing (argmax only). Dropping any one sensor costs < 0.9 pt.

### 3.5 Others 2025

- **Sumitou Atsuya (rTsfNet)**, DOI 10.1145/3714394.3756191: "axis inversion and the integration of left and right sensors separately for the arms and for the legs"; LOSO macro F1 **arm 0.6213, leg 0.5718**; public LB 0.57485.
- **VedaNam Coders (Two-stage reservoir computing)**, DOI 10.1145/3714394.3756190, open PDF https://kyutech.repo.nii.ac.jp/records/2002168: CNN spatial features + reservoir computing for short-term dynamics, "merged-limb RC" best **test macro F1 0.52888**. The same team is 7th on the 2026 LB (0.86132 on 2026-07-07).
- Only one public 2025 notebook exists (schw1eg0/testing-v0-535).

---

## 4. 3rd WEAR challenge 2026: everything public that could be found

Search coverage: WebSearch (Google), arXiv export API (`all:"WEAR" AND "activity recognition" AND (HASCA OR challenge OR VideoMAE)` -> only the 2025 papers above), GitHub repository search API (queries `wear hasca`, `wear dataset challenge`, `hasca 2026`, `wear videomae`, `3rd-wear OR wear2026 OR "wear challenge"`), GitHub user lookups for thisray / andao94 / benjaminbeaujard / kansukehabano / anhnamxtanh / gleb270 / huynhtruongtu (thisray and anhnamxtanh exist on GitHub with no HAR repos; the others 404), Kaggle kernels list via CLI, Kaggle discussion via the built-in browser, LinkedIn post by Kristof Van Laerhoven, Semantic Scholar. **No technical report, preprint, blog or code from any of the ten top-LB usernames was found.** UbiComp/ISWC 2026 companion proceedings (HASCA, Shanghai, 11-12 Oct 2026) are not online yet; the report deadline was 5 July 2026 (host post "Technical Report Deadline Passed", https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026/discussion/720908, encouraging arXiv posts). Kaggle discussion has only 3 threads (test-data redownload after a video-feature bug, 26 Apr; report deadline; a 20-day-old question about missing train .npy files).

### 4.1 Organiser facts (challenge page https://mariusbock.github.io/wear/challenge.html, Kaggle data page)

"random 1-second sliding windows from a single inertial sensor, as well as pre-extracted, frame-wise features (VideoMAEv2)"; test 12,234 windows, `(N,50,3)`, `(N,15,768)`, meta id/sbj_id/sensor_location, "All three files share the same ordering, defined by the id column"; VideoMAEv2-Base, 224x224, clip = frame i with 8 past + 7 future frames -> 768-d per frame; prizes EUR 300/150/75; top-3 must submit a 6-page report and present; one Kaggle account per team; no collaboration with the dataset authors. Timeline on the page says end 5 July 2026 but Kaggle still shows "20 days to go" (12 Oct 2026) and late entries keep landing on the LB.

### 4.2 Public code with numbers (2026)

| Source | Method | Score |
|---|---|---|
| yuma-iwamoto/WEAR2026-UEC-dx2 (GitHub, 1 commit, manifest dated 2026-07-03; UEC = Univ. of Electro-Communications, Tokyo; team name unknown) | 6-model ensemble, weights by beam search (width 8, 250 steps) on held-out sbj 18-21: Inertial-LightGBM (all hand features incl. Hjorth/FD/Welch/bandpower/lags/segments/pitch-roll-tilt; depth 5, lr 0.03, 3000 it, balanced) **w 0.272**; Inertial-CNN8 (x,y,z,|a|,|dx|,|dy|,|dz|,d|a| channels) + Video-MLP (mean/std/delta5/56 scalar projections, video-dropout 0.55) **w 0.221**; Inertial-XceptionTime (48 ch, kernel 41, 80 ep) **w 0.203**; Video-MLP on `first_mid_last` frames + deltas **w 0.157**; XceptionTime+Video-MLP w 0.138; Video-CNN w 0.009. Windows 50/stride 25, label purity >= 0.8, sensor embedding 8-d, optional left-limb canonicalisation (negate x for left arm, y for left leg), class weight sqrt/balanced, label smoothing 0.05-0.07. No temporal post-processing in the repo. | **val macro F1 0.7527** (sbj 18-21 holdout); LB unknown |
| rilink/HARMA (see 3.2) | IMU LightGBM + temporal-PCA video + DANN video softmax, stacked | LB 0.737 |
| nomannic19 "[TS/Emb] 3WDC Temporal Fusion Ensemble" (Kaggle, 11 votes, 2026-08-30) | PooledFusion (16 IMU stats + mean/std-pooled 1536-d video -> MLP) x 0.4 + TemporalFusion (Inception 1-D encoder on 4 channels + 1-layer Transformer over the 15 frames with attention pooling, gated fusion, modality dropout 0.1) x 0.6; 150 windows per subject-class; null logit bias +0.75 | LB **0.61278** (quoted by akhyar2612's notebook) |
| akhyar2612 "0.670" (Kaggle, 2026-09-18) | LightGBM on "champion" features (FAME/2024-winner style) + pooled video; modules for per-class OOF logit offsets, sensor-specialist models, domain weighting, recording selection, hierarchical family->variant (8 families), video kNN, cross-sensor propagation, temporal majority smoothing by id order | LB **0.670** (title); notes: video-only classifier 0.45; "more training data hurts (0.657 -> 0.586)"; found "~243 cross-sensor duplicate moments" and "~3050 windows per sensor" |
| honghanhhh "WEAR@HASCA" V8 (Kaggle, 2026-09-20) | Train a **4-IMU LightGBM (410 features incl. pairwise magnitude correlations + presence mask)** with sensor-dropout copies; at test, **join windows of the same subject across limbs by identical/near-identical VideoMAE fingerprints** (exact hash, rounded hash, greedy cosine >= 0.98-0.9995), rebuild (50,4,3), average probabilities inside a moment, null bias 0.35, **Viterbi per subject over moments ordered by min id, stay prob 0.93** | previous version "0.639"; V8 score not stated ("0.80 needs the join to recover ~3000 groups of 4") |
| levienbang/hasca-wear (GitHub, 2026-09-20) | 01: single-sensor XGBoost, enhanced features, val (subject holdout) **0.6693 -> 0.6968** one model, 0.6984 per-sensor models (arm ~0.704, leg ~0.684-0.696). 02: VideoMAE-only 15-token Conv1d net, null excluded: val **0.6279** best epoch (overfits after epoch 1) | no LB |
| Agnuxo1/wear-hasca-2026 | participant-aware baseline scaffold, no scores | - |
| acco-cyber/wear-hasca2026-challenge (ours) | LightGBM subject-disjoint CV, temporal-context features, null calibration | 0.71528 |

Other public 2026 kernels (lakhindarpal, hmnshudhmn24, binasalama, udaken10, stmugiwara CNN-LSTM v3/v5/v6/v7) are baselines without reported scores above ~0.6.

### 4.3 What the 0.84-0.88 teams are probably doing (inference, not evidence)

- The LB jump from the best context-free public pipeline (0.737 HARMA stacking, 0.7527 UEC val ensemble) to 0.84-0.88 by teams with 4-6 submissions (Vedanam Coders 0.861 on their first days, Gleb Shanshin 0.858, KMET 0.846 all before 7 July) is far larger than any modelling gain reported in the literature (+1-2 pt per trick) and matches the size of the **temporal-context gain** documented on WEAR (+5 pt from 10 s majority vote on 4-sensor DeepConvLSTM; multi-window features were the whole 2024 recipe). Since the test tiles every second and adjacent seconds share VideoMAE context, recovering each subject's timeline (video-feature similarity chaining / kNN graph) and smoothing per bout is the most plausible shared lever; honghanhhh's public V8 already implements Viterbi-per-subject.
- Second lever consistent with the literature: **video kNN / retrieval against the labelled train frames** (sbj_25 is at seen location 3; the 2024 protocol rule ">= 50 s per activity per participant" and the two-session grouping give strong priors for label-count matching per subject).

---

## 5. Open questions the literature does not settle

1. Whether within-subject `id` order is temporal (akhyar2612 asserts it; honghanhhh orders by min id; the context says unknown). Cheap test: autocorrelation of VideoMAE features vs id within a subject.
2. akhyar2612's "~243 cross-sensor duplicate moments" and "~3050 windows per sensor" contradict a pure one-limb-per-second tiling (12,234 = total seconds); duplicates may be near-identical adjacent seconds or genuinely repeated moments - needs a direct check with the exact-hash join from honghanhhh's notebook.
3. sbj_22 window count (5197) exceeds its summed session duration (5017 s) by 180 s - either the meta durations are rounded/trimmed or 3 minutes of extra footage exist.
4. FAME full text (sign-flip probability, PCA dims, loss weights) is unreadable (ACM Cloudflare); the repo has no code. Try the ACM PDF through the desktop browser manually or e-mail the authors.
5. Private/public split of the 2026 LB is unknown (2025 was ~50/50); all quoted 2026 numbers are public LB.

---

## 6. URL index

- WEAR paper: https://arxiv.org/abs/2304.05088 ; https://dl.acm.org/doi/10.1145/3699776 ; supplementary: https://github.com/mariusbock/wear/blob/main/supplementary_material.pdf ; site: https://mariusbock.github.io/wear/ ; challenge page: https://mariusbock.github.io/wear/challenge.html
- 2024 winner: https://arxiv.org/abs/2408.03947 (DOI 10.1145/3675094.3678453)
- 2025: FAME https://doi.org/10.1145/3714394.3756194 , https://github.com/FranciscoCalatrava/FAME-Feature-Augmented-Multi-View-Ensemble ; HARMA https://doi.org/10.1145/3714394.3756192 , https://d-nb.info/1386433594/34 , https://github.com/rilink/HARMA ; 3KA https://arxiv.org/abs/2511.23173 (DOI 10.1145/3714394.3756193), https://github.com/Khanghcmut/WEAR-Challenge ; PatchTST https://arxiv.org/abs/2510.21282 ; rTsfNet https://doi.org/10.1145/3714394.3756191 ; reservoir https://doi.org/10.1145/3714394.3756190 , https://kyutech.repo.nii.ac.jp/records/2002168 ; Kaggle 2025: https://www.kaggle.com/competitions/2nd-wear-dataset-challenge
- 2026: Kaggle https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026 (discussion ids 694742, 720908, 738942); https://github.com/yuma-iwamoto/WEAR2026-UEC-dx2 ; https://github.com/levienbang/hasca-wear ; https://github.com/Agnuxo1/wear-hasca-2026 ; Kaggle kernels nomannic19/ts-emb-3wdc-temporal-fusion-ensemble, akhyar2612/0-670, honghanhhh/wear-hasca, binasalama/wear-hasca-baseline; LinkedIn announcement https://www.linkedin.com/posts/kristof-van-laerhoven_3rd-wear-dataset-challenge-hasca-2026-activity-7449482783632277504-zljJ
