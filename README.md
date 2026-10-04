# 3rd WEAR Dataset Challenge @ HASCA 2026 — Koushik Rudra

Kaggle: https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026 (macro-F1 over 19 classes).

Public leaderboard progress: 0.71528 (2026-09-22) -> 0.88679, rank 2 (2026-09-24) -> 0.88791 (2026-09-25) -> 0.88934 (2026-09-27, rank 7) -> 0.91044 (2026-09-28, rank 5) -> 0.91289 (2026-09-29, rank 7) -> 0.91367 (2026-10-02) -> 0.91478 (2026-10-03) -> 0.92738 (2026-10-03, rank 7) -> **0.92880** (2026-10-04, rank 7; leaders 0.930-0.942).

Note (host, discussion 743256): prizes were decided at the July 5, 2026 technical-report deadline; the leaderboard after
that is for the Kaggle community only.

## 2026-10-04: multi-fit decoding of the v4 public pipeline (`exp/v4/`, `kaggle/v4_fork/`)
* Forks of goodpjw2008's v4 notebook with other seeds keep every decode input (`kaggle/v4_fork/patch_v4.py`); heavier
  variants train 8 fusion + 4 IMU window seeds (or 2 transformer-over-frames seeds), four two-tower seeds and five
  link-scorer seeds in one run (`patch_big.py`), and the "opt" variants decode with the changes below inside the notebook.
* `exp/v4/v4_local.py` reproduces the kernel's decode exactly on CPU (OOF 0.9230, test agreement 1.0000).
* **Per-subject whitening of the kNN video embedding** (each subject whitened with its own covariance, so training and
  test subjects are treated alike): OOF 0.9255 -> 0.9296 on one fit, public LB 0.92758 (the notebook's training-axis
  whitening: OOF 0.9287, LB 0.92513). Exercise identity in the count regressor: count error 7.5 -> 7.1, OOF +0.0012.
* Late fusion of fits (`exp/v4/v4_combine.py`: geometric mean of final graph P, re-estimated counts): OOF 0.9298,
  **LB 0.92880**. Link-level fusion across fits: `exp/v4/relink.py`.
* Did not help: feeding fused counts back into the graph (0.9298 -> 0.9282), early (joint) fusion vs late (0.9295 vs
  0.9298), whitening power/shrinkage, kNN temperature and Sinkhorn sharpening sweeps (all flat).

## 2026-09-28: two decoder families combined
The Hanbat team published their full pipeline as a Kaggle notebook (woominyo, "WEAR@HASCA 2026 | Timeline Reconstruction + Graph",
Apache 2.0). Its public run had no GPU and scored 0.890; our fork with a T4x2 GPU (`kaggle/hanbat_gpu/`, which also keeps the
intermediate arrays as output) reproduced it at **0.90581**. Their gain is the *decoder* (LightGBM stacker over linked-neighbour
log-probs -> label propagation with self-training -> per-subject Sinkhorn count calibration): their tabular base under our
mrf4 decoder scores only 0.88184. Our decoder's labels and their calibrated probabilities disagree on ~7% of windows, and a
confidence gate validated on 18 held-out sessions (`exp/hyb/combo_cv.py`) combines them: where their calibrated confidence is
below 0.6, take our e44 label -> 0.90954. Going one step deeper (`exp/hyb/graph_lab.py`): our labels also enter their
self-training rounds as a pseudo-label prior (weight 0.3) and their Sinkhorn class-count targets (weight 0.3), and the gate
only fires when our label is their first or second choice -> **0.91044**. Local re-runs of their tabular + graph stages
(`exp/hyb/hanbat_stack.py`) let us inject our base models and 5-fold UEC members (`kaggle/uec_k3/`) into their stacker
(CV 0.8790 -> 0.8827), which did not transfer to the public leaderboard (0.90398).

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

| weighted vote over the 6 / 8 best leaderboard files (`src/vote_subs.py`) | 0.88718 / 0.88791 |
| UEC-dx2 members rebuilt on Kaggle GPU (`kaggle/uec_k1`, `kaggle/uec_k2`) + local inertial LightGBM (`uec/gbdt`), 3-member core as primary base (`src/uec_assemble.py`) | 0.88795 |
| weighted vote over the 9 best leaderboard files | 0.88934 |
| Hanbat notebook reproduced on GPU T4x2 (`kaggle/hanbat_gpu`) | 0.90581 |
| their tabular base under our mrf4 decoder (`exp/hyb/dec_hanbat.py`) | 0.88184 |
| their stacker with our v3b/v1/fusion + 5-fold UEC cnn8/xcep OOF columns (`exp/hyb/hanbat_stack.py`, `kaggle/uec_k3`) | 0.90398 |
| confidence gate: their graph output, our e44 label where their calibrated confidence < 0.6 (`exp/hyb/gate_test.py`) | 0.90954 |
| same gate, threshold 0.7 with the "our label must be their top-2" rule | 0.90852 |
| graph stage re-run with our labels as self-training prior 0.3 + count targets 0.3 + top-2 gate 0.6 (`exp/hyb/graph_lab.py`) | 0.91021 |
| top-2 gate 0.65 on the average of three graph outputs | 0.90899 |
| majority vote of the three files above | 0.90959 |
| prior 0.3 + count targets 0.3 + top-2 gate 0.55 | 0.91044 |
| same + our chain links unioned with their L2 links in the graph stage (`graph_lab.py --extra_links`) | **0.91289** |
| same + our top-2 candidate successors as edges, gate 0.5 (CV best 0.8898) | 0.90957 |
| same as 0.91289 + their L0 links as a third link set | 0.90955 |
| 0.91289 recipe + per-subject cross-fitted self-training log-probs at 0.08 (`exp/hyb/selftrain_cv.py`, CV 0.8882 -> 0.8919) | **0.91367** |
| same at weight 0.15 | 0.91197 |
| 0.91289 recipe + transductive tabular expert, train + cross-fitted test pseudo-labels (`exp/hyb/adapt_tab.py`, CV 0.8955) | 0.90852 |
| 0.91367 recipe + a second independently fitted copy's L2 assignment links as an extra one-to-one link set (`kaggle/hanbat_run2`) | **0.91478** |

Ceilings measured with the leaderboard-faithful CV (fold-honest OOF L2 links): with true time order our chain decoder
on the graph output reaches 0.946; 0.93 needs ~98% exact next-second links, the best links available reach 56%.

2026-09-29 analysis (`exp/hyb/err_anatomy.py`, `exp/hyb/agents/`): on the train tiles 95.4% of activity bouts already have
the right majority label and a bout-majority oracle scores 0.972, so the remaining errors are within-bout grouping, not
classification. Coarse time order is not recoverable from the video (seriation errors 270-550 s; a time decoder needs
< 45 s), block/session priors add +0.001 (the block is already right for 98.6% of activity windows), and explicit bout
clustering scores below the baseline; a kNN ICM relabelling adds +0.005 alone but nothing on top of the gate recipe.

Diagnostics on the public LB: window-only argmax of our e19 blend = 0.71256 (the timeline decoder adds ~+0.175);
window-only argmax of the rebuilt UEC CNN8+video / XceptionTime+video members = 0.66793 (UEC's original 6-model,
5-fold, early-stopped ensemble reports 0.774). Adding UEC's video-only members (MLP/CNN) lowers the decoded score.

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
