# WEAR @ HASCA 2026 -- SYNTHESIS: levers, two recipes, the gate experiment, pitfalls

Lead synthesis, 2026-09-23. Inputs read in full: `kernels_A.md`, `kernels_B.md`, `kernels_C.md`, `repos_2026.md`,
`literature.md`, `data_analysis.md`, `top_teams.md` (all in `E:\Claude code\wear\research\`), plus the on-disk state
checked at write time (section 7). Competition: macro-F1 over 19 classes, public LB top 0.88425 / 0.88126 / 0.87430;
rank 3 needs >= 0.8743. Our best: 0.71528 (rank 37). ~19 days left (close 2026-10-12).

---

## 0. Bottom line

1. **The 0.84-0.88 tier is a structure exploit, not a better window classifier.** Every independent-window pipeline
   saturates at 0.70-0.77: HARMA 0.737 (0.725 honest CV), UEC-dx2 0.774 (0.7527 val, 6-model ensemble), last year's
   winners "Whatever" 0.772 after 188 submissions, akhyar's exhausted LightGBM campaign 0.657-0.670, our 0.715.
   Four September entrants reached 0.84-0.88 in 4-6 submissions (`top_teams.md` 1.3). The LB has a hard gap
   0.816 -> 0.841 (rank 11 -> 10).
2. **The exploitable structure is verified on the real test file** (`data_analysis.md`): the test tiles every second of
   every session (12,234 windows = summed session seconds; 0 overlaps, 0 duplicates), ids are a uniform shuffle (no
   temporal info), but temporal successor links CAN be rebuilt from the 15x768 VideoMAE block + the 20 ms inertial
   boundary gap: held-out-verified exact-successor precision 0.65 / 0.79 / 0.67 / 0.62 for sbj 22/23/24/25
   (video+inertial), 0.81-0.94 for the top-20 % scored links, and wrong links stay inside the same scene/bout.
3. **The wrong bet of every public kernel** was to smooth along `id` order (akhyar `temporal_smooth.py`,
   honghanhhh Viterbi by min-id) or to join limbs across windows (honghanhhh "camera moments"). Both are measured
   to be noise. The right bet is per-subject timeline reconstruction + bout-level decoding + per-subject class-count
   / null budgeting.
4. **Both recipes below share one base (a HARMA/UEC-class stacked window classifier, expected raw LB 0.74-0.77)** and
   differ in how they decode the timeline: Recipe A = ordered chains + Viterbi + count calibration (CPU, uses the
   `src\` scaffolding that already exists); Recipe B = temporal kNN graph smoothing + segment-to-class assignment
   under the "18 classes x 40-230 s per subject" constraint + transductive self-training. Expected public LB
   A: 0.84-0.88, B: 0.86-0.90.
5. **The gate experiment is `src\simulate.py`** on held-out train sessions with known order (simulate the test
   construction, rebuild chains, decode OOF probabilities): it measures exact-link precision, same-label edge rate
   and the end-to-end F1 gain raw -> chain-decoded -> oracle order. It only needs `data\prep\train_vid_pca.npy`
   (still 0 bytes, downloading) plus any OOF probability file. Run it before spending a single submission.

---

## 1. Evidence map: where the points are

| tier (public LB) | who | what they do | evidence |
|---|---|---|---|
| 0.03-0.56 | broken / inertial-only baselines | organiser DCL 0.531, A-and-D 0.561 | `top_teams.md` 1.1 |
| 0.60-0.67 | public-kernel copies | pooled video + IMU stats, no validation, blind null bias exp(0.35)-exp(0.75); nomannic 0.613 (6k windows, 3-4 epochs), akhyar 0.670 (7 LightGBMs, 600 cap), 5 accounts tied at 0.66456 | `kernels_A/B/C.md` |
| 0.70-0.74 | HARMA 0.737, us 0.715 | stacked window classifiers; HARMA: 83 IMU feats + per-subject-centred temporal video PCA + DANN LSTM + 4 meta-learners; ours: kNN-propagated LightGBM + public-vote blends | `repos_2026.md` 1, 4 |
| 0.77-0.82 | UEC-dx2 0.774, Whatever 0.772, 13 others | strong multimodal ensembles (UEC: LightGBM + CNN + XceptionTime + Video-MLP, beam-weighted, val 0.7527); still window-independent | `literature.md` 4.2 |
| 0.84-0.885 | 10 teams, 4 of them with 4-14 subs in September | no public code; the only mechanism consistent with (i) the size of the jump, (ii) 4-6 submissions, (iii) DFKI stuck at 0.77, is per-subject temporal decoding of the tiled test set | `top_teams.md` 4, `literature.md` 4.3 |

Quantitative anchors for the temporal lever (`literature.md` 1.2, 2): WEAR paper 10 s majority vote lifts 1 s
DeepConvLSTM from F1 70.36 to 75.44 (+5.1) and mAP 9.26 -> 59.14 on 4-sensor data; the 2025 challenge removed
context (shuffled windows) and the same groups fell from ~0.90 (2024, contiguous, 4 sensors) to 0.60; the 2024
winner's entire recipe was +-32 s multi-window context + Gaussian smoothing + the rule "every activity >= 50 s per
recording". A single-sensor 1 s model has a far higher error rate than the 4-sensor DCL, so smoothing has more to
remove; that is why +0.08-0.12 (not +0.05) is the expected size here.

---

## 2. Ranked levers (expected gain on the public LB, confidence, evidence)

Gains are incremental in the order listed and assume the previous ones are in place. "Base" = raw argmax of the
stacked window classifier.

| # | lever | expected LB gain | confidence | evidence | how to validate before submitting |
|---|---|---|---|---|---|
| 1 | **Per-subject timeline reconstruction + temporal decoding** (chains or kNN graph over VideoMAE tail/head similarity + inertial boundary gap; Viterbi / graph smoothing of class log-probs) | **+0.08 to +0.12** over base (0.75 -> 0.83-0.87) | high on direction (only mechanism consistent with the LB), medium on magnitude (test link precision is estimated, not ground-truthed) | `data_analysis.md` 3b: held-out precision 0.62-0.79, top-20 % 0.81-0.94, sym mean-pool NN paths cover 75-99 % of windows in paths >= 100; `literature.md` 1.2: +5.1 F1 from 10 s vote on a much stronger model; `top_teams.md` 1.3-1.4 | `src\simulate.py` on 6 held-out train sessions: f1_viterbi - f1_raw and f1_oracle_order - f1_viterbi |
| 2 | **Per-subject class-count and null budgeting** (each of 18 classes 40-230 s per subject, null = 1 - ~1830 s / recording length: sbj 22 ~55-65 %, 23 ~35-45 %, 24/25 ~15-35 %) | **+0.02 to +0.04** | high | `data_analysis.md` 5 (activity content 1553-2231 s across 10 subjects, null 0.19-0.55 rising with length); worklog: global 38 % prior was ~7 pts under the LB optimum ~45 %, d6 overshoot cost -0.035; `kernels_C.md`: LightGBM predicts 58.9 % null on test vs 39.7 % train, per-class calibration alone +0.015 OOF; 2024 winner's ">= 50 s per activity" rule | `decode.calibrate_counts` inside the simulation; tune lo/hi and the null budget on train sessions of matching length (sbj_3/sbj_2 ~ sbj 22; sbj_0/1/7 ~ sbj 23; sbj_5/9 ~ sbj 24/25) |
| 3 | **Base classifier to HARMA/UEC level**: stride-1 s tiles with purity >= 0.8, 100-d FAME IMU block per limb, per-subject-centred temporal video PCA, fusion net over the 15x768 frames, 6-fold subject-disjoint stacking, null undersampled to 3x median, 3 seeds | **+0.03 to +0.06** over our raw 0.615-0.715 (base 0.74-0.78) | high | HARMA README 0.591 -> 0.663 -> 0.683 -> 0.725 CV / 0.732 LB -> 0.737; UEC val 0.7527 / LB 0.774; akhyar 0.657 flat -> 0.670 hierarchical; our row-F1 0.62 came from first-sample labels, even windows only, pooled video (`repos_2026.md` 4) | 6-fold subject-disjoint OOF single-limb macro-F1 target >= 0.72 (HARMA's 0.725 CV tracked LB within 1 pt) |
| 4 | **Transductive per-subject adaptation**: standardise IMU features per (test subject, limb) and video PCA per subject using test sbj_id; optional DANN with test subjects as unlabeled domains; 1-2 rounds of self-training on decoded pseudo-labels | +0.01 to +0.03 pre-decode, larger after decode (it removes the systematic per-bout confusions that smoothing cannot) | medium | HARMA's per-subject centring is inside its +7 pt step (0.591 -> 0.663; the isolated share is not reported); `repos_2026.md` 5d: top-1 video neighbour is same-subject 99.7 % (strong subject/scene shift); test null over-prediction | simulate with the held-out session treated as the "test subject" (centre on it, self-train on its decoded labels) |
| 5 | **Chain-neighbour multi-limb context**: pool other-limb IMU features from +-5 chain neighbours (train with the same simulated pooling) | +0.01 to +0.03 | medium-low (overlaps with lever 1) | WEAR supp. Table 9: right wrist only F1 61.97 vs 4 sensors 75.44; 2024 UL-pairing +1.9; `data_analysis.md` 6 | simulation only; skip if lever 1 already saturates |
| 6 | Family->variant hierarchy + 0.7/0.3 flat blend; orientation augmentation (channel sign-flip, +-15 deg rotation, L/R mirror); seed bagging | +0.01 to +0.02 each | medium-high | akhyar 0.657 -> 0.670; FAME: "frequency features and sign-flipping were the main drivers"; PatchTST +1.6 with test-matched aug; 2024 LR-swap +1.3 | OOF |
| 7 | Sequence model over the 15 frames instead of mean/std pooling (in isolation) | +0.02 to +0.03 | medium | HARMA pooled PCA 0.663 vs LSTM-DANN 0.683; levienbang video-only Conv1d val 0.628; nomannic 0.613 undertrained | OOF; already part of lever 3 (`kaggle\fusion\fusion.py`) |

Negative / zero levers (measured or reasoned, do not spend time on them): smoothing or Viterbi along `id`
(consecutive-id video cosine equals random pairs in all 4 subjects, `data_analysis.md` 2) = 0 or negative;
cross-limb "camera moment" joins (0 duplicate video windows) = 0; video-kNN against TRAIN for label histograms or
the null prior (cross-subject lookup, underestimated null 38 % vs 45 %) = -0.02 to -0.04; distilling public votes
into our model (d4 < d3) and adding honghanhhh votes (d5 < d3) = negative; OOF-tuned per-class logit offsets on the
22-subject pool (akhyar: "frequently overfit the held-out subjects and hurt the real test set").

Sum check: 0.715 (ours) -> lever 3 -> 0.74-0.78 raw -> lever 1 -> 0.83-0.87 -> lever 2 -> 0.85-0.89 -> levers 4-6 ->
0.86-0.90. The band overlaps the observed top tier (0.84-0.885), which is the sanity check that the plan is
consistent with what 10 teams actually achieved.

---

## 3. Recipe A -- "Chain-Viterbi": stacked window classifier + LSA timeline + Viterbi + count calibration

Designed to run mostly on the local CPU box (12 cores / 34 GB) with the code that already exists in
`E:\Claude code\wear\src\`; the Kaggle GPU is used once for the fusion net.

### A.1 Data prep (done or in flight)
- Use the Kaggle prep kernel outputs in `E:\Claude code\wear\data\prep\` (`kaggle\prep\prep.py`): every train
  session tiled at stride 1 s in the exact test format: `train_meta.csv` (69,326 seconds: session, sbj, t, y =
  majority label, y_c, pur = purity, n_nan_limbs), `train_imu.npy` (N,4,50,3) float16, `train_vid_pca.npy`
  (N,15,160) float16 = frames [30t+8, 30t+23) projected by PCA(160) fit on train frames, `train_vid_mean768.npy`,
  `test_vid_pca.npy` (12234,15,160), `test_vid_mean768.npy`, `pca_*.npy`. Alignment verified: sbj_0 has 139,725
  samples = 2794.5 s and 83,835 frames = 2794.5 x 30 (`kaggle\probe\out\wear-probe.log`).
- Training rows = (second, limb) with the limb non-NaN and `pur >= 0.8` (HARMA: majority > 40/50; UEC: 0.8).
  Optional 2x data: rerun prep with a 25-sample offset (stride 0.5 s like akhyar/HARMA) -- not required.
- Folds: 6 subject-disjoint folds, seed 42, `sbj_0_2` -> subject 0, `sbj_14_2` -> subject 14 (same partition as
  HARMA so OOF numbers are comparable). Write the partition to `work\folds.json` and use it in every model.
- Null: per fold, undersample train-side null rows to 3x the median activity count (HARMA; akhyar's "more data
  hurts 0.657 -> 0.586" is this null-domination effect); class weights 1/sqrt(freq) clipped [0.5, 3].

### A.2 Base models (target OOF single-limb macro-F1 >= 0.72)
1. **LightGBM-A (CPU)**: per row, 100-d FAME block on the limb's (50,3) window (akhyar `features.py`
   `extract_acc_features_vect`: x,y,z,|a| x [13 time + 8 spectral + 4 half-window]; extracted copy in the
   scratchpad `akhyar\cell2_features.py` -- copy into `src\imu_feats.py`) + limb one-hot + video: mean/std over the
   15 PCA-160 frames (320), first/mid/last frame deltas (UEC), and a 64-d **temporal PCA of the flattened 15x160
   block fit on train+test and mean-centred per subject including the 4 test subjects** (HARMA `base_models.py`).
   Params: 800 rounds, lr 0.05, 63 leaves, min_child 30, colsample 0.7, subsample 0.8, lambda 1; 3 seeds.
   Add the 8-family model + 5 variant models, blend 0.7 hierarchical / 0.3 flat (akhyar +0.013).
2. **Fusion net (Kaggle T4x2)**: `E:\Claude code\wear\kaggle\fusion\fusion.py` as drafted (Conv1D IMU on 7 channels
   + 2-block Transformer over the 15x768 frames with CLS + limb embedding, modality dropout 0.15, aux heads, label
   smoothing 0.05, null 3x median, sqrt class weights), changed to the shared 6-fold partition, 12-15 epochs, 2 seeds.
   Emits `oof.npy` (N,4,19) and `test.npy` (12234,19).
3. **Stack**: multinomial logistic regression on concatenated log-probs (LGBM flat, LGBM hier, net) fit on OOF ->
   calibrated P_train (N,4,19) / P_test (12234,19). Expected: LGBM-A ~0.66-0.69, net ~0.66-0.70, stack 0.72-0.75
   (HARMA 0.725 with a weaker net; UEC 0.7527 with 6 members).

### A.3 Timeline reconstruction (per test subject)
- `src\chain.py`: 9 pair features per (window, top-40 candidate successor): tail3/head3 cosine, last/first cosine,
  mean-pool cosine, row/column dominance margins, same-limb flag, log1p inertial boundary gap and linear-extrapolation
  gap (same limb only). Logistic scorer fit on >= 12 train sessions with known order (simulated test construction:
  random limb per second, shuffle). Linear assignment on the log-odds, drop edges below `min_logodds` (sim-tuned),
  cut the weakest edge of every cycle -> chains; keep each edge's log-odds as confidence.
- Quantise test inertial to float16 before scoring (train pairs come from float16 `train_imu.npy`; test is float64)
  and check that PCA-160 descriptors on test reproduce the raw-768 statistics of `data_analysis.md` 3a (best-match
  cosine ~0.90-0.93, best-vs-second margin ~0.010, mutual-best 42-49 %).
- Sanity targets on test (from `research\artifacts\chain_heldout_summary.csv`): same-limb edges with z-gap below
  2x intra-window step 0.45-0.51 (random 0.055-0.084); nodes in chains >= 100: 8-31 %.

### A.4 Decoding
1. `decode.viterbi_chains` along every chain with p_stay 0.95 (sim-tuned in 0.90-0.98), temperature 1, and the
   off-diagonal transition mass scaled by edge confidence (log-odds < 0 -> near-uniform transition, i.e. a soft
   chain break) -- implement as a per-edge p_stay.
2. `decode.calibrate_counts` per subject: additive log-biases iterated until every activity class has 40-230 windows
   after decoding (train: every activity 62-225 s per subject, `data_analysis.md` 5).
3. Per-subject null budget: bisection on the null log-bias so that the decoded null fraction lands in the band
   predicted from recording length (22: 55-65 %, 23: 35-45 %, 24/25: 15-35 %); the exact target inside the band is
   tuned in the simulation on train sessions of matching length.
4. Short-segment clean-up: runs < 5 s of a non-null class inside a chain are merged into the neighbour (train label
   glitches are < 3 s).

### A.5 Validation protocol (subject-disjoint, protocol-matched)
- Level 1: 6-fold subject-disjoint OOF single-limb macro-F1 (one random limb per second, like test).
- Level 2 (the one that matters): `src\simulate.py --oof work\stack\oof.npy` on held-out sessions
  (default eval `sbj_0, sbj_5, sbj_10, sbj_14_2, sbj_20, sbj_21`; add `sbj_3` (69.5 min, null 0.55 ~ sbj 22) and the
  concatenation `sbj_0 + sbj_0_2` as a two-session subject). Report per session: f1_raw, f1_viterbi, f1_calib,
  f1_oracle_order, f1_oracle_calib, edge precision, same-label edge fraction, chain-length distribution.
  Only OOF probabilities of the held-out session are used (no leakage), the scorer is fit on the other sessions.
- Go/no-go for a submission: mean(f1_calib - f1_raw) >= +0.06 and no session regresses.

### A.6 Submission plan (3 submissions to locate ourselves)
1. raw stack argmax (expected 0.74-0.77; if < 0.72 the base is the problem, fix it before decoding);
2. chain-Viterbi decoded (expected +0.06-0.09);
3. + count calibration and per-subject null budget (expected +0.02-0.03).

### A.7 Expected public LB: **0.84-0.88** (median 0.86)
Reasoning: base 0.75 +- 0.015 (HARMA 0.737 / UEC 0.774 bracket it); decode +0.07 +- 0.02 (single-sensor model has
~25-30 % window error, chains cover most windows, links 62-79 % exact and the rest same-bout; WEAR's +5.1 was on a
model with far fewer errors to remove); counts/null +0.02 +- 0.01. Reaches >= 0.875 when the base is >= 0.76 AND the
simulated decode gain is >= +0.09.

### A.8 Risks
- Test link precision below the train simulation (new scenes; sbj 25 is the weakest at 0.62) -> smaller gain; the
  per-edge confidence keeps damage bounded (weak edges decode like isolated windows).
- Systematic per-bout confusions (a whole stretching bout read as another stretching variant) are not fixed by
  smoothing; that is Recipe B's assignment step.
- Wrong null band for sbj 24/25 (sets were shortened: 1995/1914 s cannot hold 1830 s of activity at train pacing).
- sbj 22 has 180 windows more than its listed sessions (5197 vs 5017): budget on 5197.

---

## 4. Recipe B -- "Graph-Assign": temporal kNN graph smoothing + segment-to-class assignment + transductive self-training

Shares A.1-A.2 (same prep, same base models, same folds) and adds robustness to the ~1/3 wrong links and an explicit
attack on the class-level confusions. Needs Kaggle GPU for the net and self-training rounds; the graph/assignment
steps are CPU (seconds per subject).

### B.1 Transductive base
- Per-(test subject, limb) standardisation of the IMU feature block and per-subject centring of video PCA / temporal
  PCA using test `sbj_id` (HARMA's only domain-adaptive step, inside its +7 pt).
- Fusion net with a subject-adversarial head (gradient reversal) where the 4 test subjects are additional unlabeled
  domains (HARMA `adversarial_encoder.py` pattern; DANN lambda ramp 2/(1+e^-10p)-1).

### B.2 Temporal kNN graph (per subject)
- From `chain.pair_features` (TOPM 40): keep the top-k (k = 10) successor and predecessor candidates per window
  with edge weight w = sigmoid(log-odds) from the same logistic scorer; symmetrise; add a second edge set from
  mean-pool video similarity (same-scene, `data_analysis.md` 3a: sym mean-pool NN paths cover 75-99 % of windows).
- Graph smoothing of log-probabilities: logP <- (1 - a) logP0 + a * D^-1 W logP, a = 0.5, 5-10 iterations (label
  propagation without committing to an order; tolerant to the 20-38 % wrong links because each node has 10-20
  neighbours, not 1).

### B.3 Segmentation
- Threshold the graph at log-odds > tau (sim-tuned) and take connected components, or spectral clustering on W +
  mean-pool similarity -> segments (expected 40-100 per subject including null segments). Each segment gets the
  summed smoothed log-prob vector and its size in seconds.

### B.4 Assignment under the protocol prior (ILP per subject, CBC via pulp; solves in seconds)
- Variables x[j,c] in {0,1}: segment j -> class c. Constraints: one class per segment; for each activity class c,
  40 <= sum_j size_j * x[j,c] <= 230 (soft: penalty outside the band); at most 4 segments per activity class (train:
  1-4 sets, `data_analysis.md` 5); null total within the per-subject band of A.4.3.
- Objective: sum_j sum_c x[j,c] * logP_j(c) + lambda * (same-class bonus for graph-adjacent segments).
- This is the rigorous form of the 2024 winner's ">= 50 s per activity per recording" rule and directly targets the
  classes where macro-F1 is lost (OOF F1 0.31-0.49 for stretching triceps/lunging/shoulders, lunges, bench-dips in
  `kernels_C.md`): two regions cannot both be "stretching (triceps)".

### B.5 Transductive self-training (1-2 rounds, sim-validated)
- Windows in segments with assignment margin >= 0.8 become pseudo-labels for the test subject; fine-tune the net
  (and/or retrain LGBM) with test rows at weight 0.3; re-predict, re-smooth, re-assign. Stop if the simulated
  held-out session's F1 does not improve after round 1.

### B.6 Validation
- Same `simulate.py` framework extended with graph smoothing, segmentation, assignment and self-training stages;
  report per-session F1 for each stage and per-class F1 for classes 6-10 and 18 (the weak ones) before/after
  assignment. Additional check: the number of segments per activity class produced on held-out sessions vs the
  known 1-4 sets.

### B.7 Submission plan
4. graph-smoothed argmax (should be >= Recipe A sub 2);
5. + assignment (expected +0.02-0.04 over sub 4);
6. + self-training (expected +0.01-0.02). Final choice between A and B by simulation first, LB second.

### B.8 Expected public LB: **0.86-0.90** (median 0.875)
Reasoning: same base; graph smoothing captures the same +0.07 as chains but degrades more gracefully on the
20-38 % wrong links; the assignment adds +0.02-0.04 by resolving the variant confusions at bout level (this is the
part of the gap that per-window models cannot close: the WEAR paper's video-only models "struggle to differentiate
between different running styles ... normal and complex sit-ups"); self-training closes part of the subject shift
(HARMA's centring alone was worth several points at window level).

### B.9 Risks
- ILP band wrong for a subject who skipped a class or shortened sets (train sbj_2 lacks one class; sbj 24/25 are
  short) -> use soft penalties, never hard equality.
- Pseudo-label confirmation bias -> confidence gating + the simulation must show a gain first.
- Engineering time: B.4-B.5 are 3-5 days of work; do A first, B on top of A's artifacts.

---

## 5. The single most important experiment to run first (de-risk the whole plan)

**Run the protocol simulation on train sessions with known order** -- `E:\Claude code\wear\src\simulate.py` -- as
soon as `E:\Claude code\wear\data\prep\train_vid_pca.npy` finishes (0 bytes at write time; `kaggle kernels output
koushikrudra/wear-prep-windows -p data\prep`, PID 20956, is pulling it; never kill python by name).

Steps:
1. `python "E:\Claude code\wear\src\simulate.py"` (chain-only mode): fits the logistic link scorer on
   `sbj_1,3,7,12,16,19`, simulates the test construction on `sbj_0,5,10,14_2,20,21` (random limb per second,
   shuffle), rebuilds chains, prints exact-link precision/recall, same-label edge fraction, chain count, max chain,
   fraction of windows in chains >= 50. Expectation from the test-side estimates: precision 0.6-0.8 (video+inertial),
   same-label >= 0.90.
2. Produce a quick OOF (`src\train_lgbm.py`, 200-round LightGBM on the prep features, 6-fold subject-disjoint; ~20
   min) and rerun with `--oof work\lgbm_v1\oof.npy` to get, per held-out session: f1_raw, f1_viterbi, f1_vote,
   f1_calib, f1_oracle_order, f1_oracle_calib, true null fraction.
3. Add two cases: `sbj_3` (69.5 min, null 0.55, the sbj 22 analogue) and the concatenation `sbj_0 + sbj_0_2` (a
   two-session subject, tests session-boundary behaviour).
4. Check descriptor equivalence: tail3/head3 best-match cosine, margin and mutual-best rate on `test_vid_pca.npy`
   (PCA-160) vs the raw-768 numbers in `data_analysis.md` 3a; and the float16 quantisation of test inertial.

Decision rules:
- precision >= 0.6 and same-label >= 0.9 -> chains behave as on test; proceed with Recipe A.
- mean(f1_viterbi - f1_raw) >= +0.06 and (f1_oracle_order - f1_viterbi) <= 0.03 -> chains are sufficient.
- oracle gap > 0.05 -> ordering is the bottleneck -> prioritise Recipe B's graph smoothing.
- (f1_oracle_order - f1_raw) < 0.05 even with perfect order -> base errors are systematic per bout -> prioritise
  Recipe B's assignment (B.4) and the base classifier, not the chaining.
- The observed gain in this simulation IS the expected LB gain (the test subjects are the same kind of unseen
  subjects); do not submit a decoded file whose simulated gain is < +0.06.

---

## 6. Pitfalls seen in previous attempts (ours and public)

1. **Smoothing along `id`** (akhyar `temporal_smooth.py`, honghanhhh Viterbi by min-id, stay 0.93): ids are a
   uniform shuffle across subjects and time (runs test 8635 vs 8602 expected; consecutive-id video cosine equals
   random pairs in all 4 subjects). Pure noise; can only hurt.
2. **Cross-limb "camera moment" joins** (honghanhhh V8, akhyar `run_propagate.py` "~243 duplicates"): 0 duplicate
   video windows, 0 duplicate frames, one random limb per second. The 243 near-duplicates are adjacent seconds of
   static activities.
3. **Video-kNN against train** for label histograms and the null prior (our v9): top-1 video neighbour is the same
   subject 99.7 % of the time, so a train lookup is cross-subject and under-estimated null (38 % vs ~45 % optimum);
   the per-subject null re-match inherited the bias and d6 overshot (-0.035).
4. **Weak base construction** (ours): label = first sample of the window, no purity filter, only even windows,
   video pooled to mean/std and the raw frames deleted by `proc_video.py`, 200 rounds at lr 0.12 -> row-F1 0.62.
   Fixed by the prep kernel (majority label + purity, all seconds, 15 PCA frames kept).
5. **"More data hurts"** (akhyar 0.657 -> 0.586 uncapped) is null domination + subject overfit, not harmful data:
   undersample null to 3x median, keep purity >= 0.8, subject-disjoint early stopping; do not cap activities.
6. **Train/test video mismatch** (abhinavm2811: 15 frames linspace over the whole second; udaken: all 30 frames):
   use the central 15, [30t+8, 30t+23) (nomannic/akhyar/prep) or +7..+21 (HARMA) -- 1-frame ambiguity, both fine.
7. **CV leaks**: lakhindarpal groups `sbj_0` and `sbj_0_2` into different folds; abhinav's best_iteration ~55 at lr
   0.05 on 1536 raw video dims = immediate subject overfit. Group repeated sessions with their subject; compress and
   centre video features.
8. **Blind null multipliers** (exp(0.75) nomannic/akhyar vs exp(0.35) honghanhhh) and a **global null prior**: the
   true per-subject null fraction spans ~15 % to ~65 % on this test set. Budget per subject.
9. **Blending public votes** was our strongest lever (+0.07) but also a trap: distilling them into our model
   reduced diversity (d4 < d3), honghanhhh votes hurt (d5), and nothing in that family exceeds ~0.75.
10. **OOF-tuned per-class offsets** on the 22-subject pool did not transfer (akhyar docstring; our +5 pt OOF from
    propagation did not show on the LB): the test pool is 4 subjects x thousands of same-scene windows, a different
    regime. Validate post-processing with the protocol simulation (per held-out session), not pooled OOF.
11. **Using the LB as the validation set** (every public kernel). We have ~19 days; the simulation is cheaper and
    private-LB-safe (split unknown; a per-subject joint decode uses all windows so it transfers).
12. **"Exactly one contiguous bout per activity"** (harness context and several reports) is wrong on train: each
    activity is a compact region of 1-4 sets (~30 s) with 10-40 s rests, total 62-225 s per subject; some subjects
    interleave sets of two exercises (sbj_8). Decoders must allow several segments per class (<= 4) and never force
    one segment.
13. **Numerical mismatch between train prep (float16 IMU, PCA-160 video) and test (float64, raw 768)** for the link
    scorer: quantise test to float16 and project test video with the same PCA before scoring (already done for video
    in `test_vid_pca.npy`).
14. **Process hygiene**: a workflow agent once ran `Stop-Process -Name python` and killed the downloads; two python
    processes are live now (downloader PID 22472, kernel-output pull PID 20956). Never kill by name.
15. **Credential hygiene**: the task text contained a GitHub personal access token in plain text; it was not used by
    the research agents and should be rotated before anything is pushed to
    https://github.com/acco-cyber/wear-hasca2026-challenge.

---

## 7. Asset inventory at write time (2026-09-23 00:30)

- Test set complete: `data\test\test_inertial_data.npy` (12234,50,3), `test_videomae_data.npy` 1,127,485,568 bytes
  (12234,768,15), `test_meta_data.csv`.
- Train inertial CSVs: 12 of 24 on disk (`sbj_0..sbj_11`; `sbj_12.csv` 0 bytes, downloading); no
  `train\videomae_feat\` yet (downloader PID 22472 running `download_data.py`).
- Kaggle-side prep already done (`kaggle\prep\prep.py`, kernel `koushikrudra/wear-prep-windows`): in `data\prep\`
  `train_meta.csv` (69,326 s), `train_imu.npy` 83 MB, `train_vid_mean768.npy` 106 MB, `test_vid_pca.npy`,
  `test_vid_mean768.npy`, `pca_*.npy` downloaded; **`train_vid_pca.npy` 0 bytes (last file, in flight)**.
- Code: `src\build_windows.py`, `imu_feats.py`, `train_lgbm.py`, `predict.py`, `chain.py` (scorer + LSA),
  `decode.py` (Viterbi, chain vote, count calibration, segments), `simulate.py` (the gate); Kaggle kernels
  `kaggle\fusion\fusion.py` (5-fold fusion net, ready), `kaggle\probe\`.
- Research artifacts: `research\artifacts\chain_successor_video_xyinertial.csv` (test successor links, all 4
  subjects), `chainv2_succ_sbj*_{video_only,video_inertial}.npy`, `chain_heldout_summary.csv`, `video_meanpool.npy`
  (12234x768), `train_label_runs.csv`; scripts `research\scripts\01..07_*.py`.
- Previous pipeline: `acco\scripts\*.py`, submissions `acco\download\sub_d1..d7.csv`, `acco\worklog.md`.

Open items that the gate experiment settles: exact-link precision on known order; same-label edge rate;
the decode gain vs oracle; descriptor equivalence PCA-160 vs raw 768. Still open afterwards: the true null bands for
sbj 24/25 (short sessions); whether sbj 22's extra 180 s form a separate chain; the +7 vs +8 frame offset (cosmetic).

---

## 8. 19-day schedule

- Days 1-2: gate experiment (section 5); LightGBM-A with the 100-d FAME block + temporal PCA on the prep features;
  fold partition file. Submission 1 (raw stack) only if OOF >= 0.70.
- Days 3-5: fusion net on Kaggle (6-fold, 2 seeds); stack; Recipe A decode; submissions 2-3.
- Days 6-10: Recipe B graph smoothing, segmentation, ILP assignment; submissions 4-5.
- Days 11-14: self-training round, chain-neighbour multi-limb context, hierarchy/augmentation/seeds; submission 6.
- Days 15-19: reserve; final selection by simulation first, LB second; push code + report to the GitHub repo
  (after rotating the leaked token).
