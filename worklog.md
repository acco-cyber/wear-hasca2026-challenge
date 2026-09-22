---
Task ID: 1
Agent: Super Z (main)
Task: Recover from environment reset, rebuild WEAR HASCA 2026 pipeline, use remaining submissions to maximize score, push to GitHub.

Work Log:
- Environment was completely wiped (no kaggle CLI, no data, no scripts). Reinstalled kaggle CLI, restored token.
- Recovered submission history from Kaggle API: best prior = sub_v8 (0.71528); the 3 failed subs mentioned by user = c1 0.630 / c2 0.646 / c3 0.591.
- Re-downloaded all data (test set, 24 train inertial CSVs, 24 train videomae npys stream-processed to pooled features, test videomae pooled).
- Decoded data structure from first principles:
  * Windows = 1 second. Test: 1 window/sec proven by session durations (sbj_23 3128s=3128 win, sbj_24 1995, sbj_25 1914).
  * Train CSV = 50Hz continuous; official windows = 50 samples @ 50Hz, stride 25 (verified via akhyar2612 public kernel config).
  * Train video npy = flat 30fps frames; center-15 per window (offset +8..+22).
  * Official label mapping extracted from public kernels (null=0 ... bench-dips=18).
- Built limb-matched single-sensor pipeline: test windows carry sensor_location matching the 4 train views exactly. 554,528 (window,limb) rows x 105 IMU features + video PCA-48 + fold-safe video-kNN label histograms (40 feats).
- Trained 4-fold subject-disjoint LGBM (row-F1 ~0.62); decode: propagation smoothing + per-subject null-rate matching to video-kNN prior (OOF +5pt).
- SUB d1 = 0.61462. SUB d2 (self-training r1) = 0.61841. SUB d3 (blend with akhyar public 0.670 votes x0.8) = 0.69065. SUB d4 (round-2 self-train + hon2) = 0.68086. SUB d5 (+hon2 isolation) = 0.68418 (hon2 hurts). SUB d6 (null rematch overshoot) = 0.65535. SUB d7 (+udaken10 x0.25) = 0.69707 = today's best.
- Key findings: (1) per-subject null matching via video-kNN works OOF (+5pt); (2) video-kNN null prior UNDERESTIMATES test null (38% vs optimum ~45%); (3) blending decorrelated public voters is the strongest lever; (4) distilling aka labels into my model reduced blend diversity (d4 < d3).

Stage Summary:
- Today's best: sub_d7 = 0.69707 (prior best this session lineage: v8 = 0.71528).
- All code in /home/z/my-project/scripts/; submissions in /home/z/my-project/download/.
- Next steps for future sessions: GPU deep model (sensor-dropout fusion), per-class vote weights, stronger kNN (FAISS, IMU-space), family hierarchy.
