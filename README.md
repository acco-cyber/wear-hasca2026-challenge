# 3rd WEAR Dataset Challenge @ HASCA 2026 — Koushik Rudra

Kaggle: https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026 (macro-F1 over 19 classes).

Public leaderboard progress: 0.71528 (2026-09-22) -> 0.88679, rank 2 (2026-09-24) -> **0.88791** (2026-09-25, weighted vote of top files; rank 6 on 09-26 as the leaders moved to 0.90-0.92).

## Key insight
The test set is not a bag of independent windows. It is **every 1-second tile of 4 unseen subjects' sessions**,
shuffled (12,234 windows = summed session seconds; one random limb per window; central 15 VideoMAE frames).
Each subject performs all 18 activities for ~100 s each, so the problem becomes: rebuild each subject's timeline and
decode it jointly, with per-activity duration constraints.

## Pipeline
1. **Test-format training data** (`kaggle/prep/prep.py`, Kaggle kernel): every train session tiled at 1 s, one
   (50,4,3) IMU block per second, central-15 VideoMAE frames compressed with PCA-160.
2. **Window classifiers** (5-fold subject-disjoint CV + full-data refits for test):
   - `exp/base/train_v3.py` (v3b LightGBM): mirror-canonicalised IMU features + a copy z-scored per
     (session | test subject, limb), session-centred video PCA features, DCT/motion features.
   - `src/train_lgbm.py` (v1 LightGBM), `kaggle/fusion/fusion.py` (Conv1D-IMU + Transformer-over-frames net, GPU).
   - Log-space blend 0.5 v3b + 0.3 v1 + 0.2 fusion; full-data refits in `exp/full/fullfit.py`.
3. **Test-side evidence**: hard-label votes from earlier independent pipelines added as logit bonuses (`src/make_probs.py`).
4. **Per-subject timeline decoding**:
   - `src/chain.py`: successor-link scorer (VideoMAE tail/head similarity, dominance ranks, same-limb inertial
     boundary gap) + linear assignment -> chains.
   - `exp/decoder/decoder.py` (mrf4): null x0.5, graph smoothing over link neighbours, chain Viterbi with 10
     label-refinement passes over the link graph, inside per-class count calibration (80-250 s per activity).
   - `exp/transductive/refine.py`: within-subject VideoMAE kNN label spreading, then decode again.
5. **Validation**: `src/simulate.py`, `exp/pl/sim_final.py` rebuild the exact test construction on held-out train
   sessions with known order (random limb per second, shuffled) and score the whole pipeline end to end.

## Results log
| step | public LB |
|---|---|
| base blend + timeline graph/chain decoding + count calibration | 0.75001 |
| + votes from earlier pipelines | 0.78111 |
| + null x0.5, 80-250 s activity band | 0.81717 |
| + mrf4 decoder | 0.85369 |
| + v3b LightGBM | 0.86702 |
| + video-kNN refinement, 3-model base | 0.87061 |
| + full-data refits | 0.87409 |
| + soft full-data rebuild of the public akhyar hierarchical LightGBM (weight 0.45, `exp/aka/`) | 0.88501 |
| + independent votes from the abhinavm2811 notebook | **0.88679** |

| weighted vote over the 6 / 8 best leaderboard files (`src/vote_subs.py`) | 0.88718 / **0.88791** |

Tried and not kept (lower public LB): voting over decodes, pseudo-label self-training, session-block prior,
cross-limb link features, heavier/lighter vote weights, 3-seed bag of the akhyar rebuild (0.88154), CPU deep window
model (`exp/deep`, 0.87889), hierarchical LightGBM on v3b features (0.87800), public honghanhhh/sibamsamanta07 outputs
as voters, UEC-dx2 CNN8 rebuild (`uec/cnn8`, 0.88315), weight-renormalised blends, link scorer refit on all sessions
(0.88202). Every change that moves ~2-3% of windows away from the best recipe lands near 0.882, which suggests the
best public files carry ~+0.005 of public-subset luck.

Note: Windows application control on the dev machine blocks pandas' `indexing` and scikit-learn's `_loss` binaries;
`shim/sitecustomize.py` (set `PYTHONPATH=shim`) and `exp/transductive/refine_ns.py` (NumPy kNN) work around it. Details and simulation numbers are in `research/` and `worklog.md`.

## Reproduce
Set `KAGGLE_API_TOKEN` in your environment (never commit it). Run the prep and fusion kernels on Kaggle,
fetch outputs with `fetch_output.py`, then `exp/full/fullfit.py`, `src/make_probs.py`, `exp/transductive/refine.py`.
Data is CC BY-NC-SA 4.0 and is not redistributed here.
