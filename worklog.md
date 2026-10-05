# Worklog — 3rd WEAR Dataset Challenge @ HASCA 2026

## 2026-09-22
Previous pipeline (limb-matched LightGBM + pooled video + video-kNN + public-kernel votes): best public 0.71528.

## 2026-09-23
- Research sweep (public kernels, 2026 repos, WEAR paper and past winners, test-data analysis, top-team intel): `research/`.
- Found the test structure: every 1-s tile of 4 unseen subjects' sessions, shuffled; one random limb per window.
- Kaggle prep kernel tiles train sessions in the exact test format; link scorer + chain reconstruction; simulation
  harness on held-out train sessions with known order.
- Submissions: 0.750 (timeline decode) -> 0.781 (votes) -> 0.817 (null/count calibration) -> 0.854 (mrf4 decoder)
  -> 0.867 (v3b LightGBM) -> 0.8706 (3-model base + video-kNN refinement). Not kept: decode voting (0.8669),
  pseudo-label self-training (0.8659), session-block prior (0.8705).

## 2026-09-24
- Full-data refits instead of 5-fold averages: 0.87409.
- Vote weight of the strongest earlier pipeline: 1.6 -> 0.87265, 0.6 -> 0.85784 (keep 1.0).
- Cross-limb continuity link features: +0.008 in simulation but 0.86504 on the leaderboard (dropped).
- Activity-order / session-block decoders: flat in simulation (not submitted).
- Soft full-data rebuild of the public akhyar hierarchical LightGBM as a blend member: 0.25 -> 0.88342,
  0.45 -> 0.88501, 0.7 -> 0.88355.
- + independent votes (abhinavm2811 notebook): 0.88679 (public rank 2).
- 3-seed bag of the akhyar rebuild instead of the single seed: 0.88154 (seeds agree only ~84%; single-seed gain partly luck on the public subset).

## 2026-09-26 / 27
- Full-data GPU refit of the fusion net (3 seeds): 0.88494 (5-fold average was 0.88679).
- Window-only diagnostic of the e19 blend: 0.71256; the timeline decoder adds ~+0.175.
- Rebuilt the public UEC-dx2 ensemble: K1 GPU kernel (CNN8+video, XceptionTime+video), K2 GPU kernel
  (XceptionTime inertial, video MLP, video CNN), local inertial LightGBM (tile rows, lr 0.08, 900 iters).
  Window-only LB of the K1 pair: 0.66793 (original ensemble 0.774; full-data fixed-epoch fits are weaker than
  UEC's 5-fold early-stopped fold averages). 3-member core (LightGBM, CNN8+video, XceptionTime+video) as primary
  base + v3b + akhyar rebuild + votes: 0.88795 (best single recipe). All 6 members: 0.88160; 4 inertial-carrying:
  0.88673; core added to e19 at 1.0: 0.88418; core alone + votes: 0.87365.
- Weighted vote over the 9 best files: 0.88934 (public rank 7; leaders 0.905-0.930).

Lesson: on this test set (new subjects, new locations, different camera) the leaderboard rewards independent model
evidence; decoder/link tweaks that only re-process our own predictions do not transfer below ~+0.01 simulated gain.

## 2026-09-28 (5 submissions)
- Found the Hanbat team's public notebook (woominyo, Apache 2.0; their LB 0.903, public CPU run 0.890). Forked it to a
  GPU T4x2 kernel that keeps every intermediate array (`kaggle/hanbat_gpu/patch_nb.py`): window blend OOF 0.7234, 50 min.
  Reproduction: 0.90581. Their OOF tiles are exactly our 69,326 train rows (`exp/hyb/align_rows.py`).
- Their tabular base under our decoder: 0.88184 -> their graph propagation + Sinkhorn decoder is the gain, not their base.
- `kaggle/uec_k3`: 5-fold UEC CNN8+video / XceptionTime+video with OOF tile predictions (45 min GPU); OOF window F1
  0.679 / 0.691 on their tiles (our v3b 0.652). Injected with our v3b/v1/fusion into their S3 stacker
  (`exp/hyb/hanbat_stack.py`): S3 OOF 0.8116 -> 0.8246, graph CV 0.8790 -> 0.8827, LB 0.90398 (no transfer).
  K3 members in the window blend: CV 0.8798; our OOF base mixed into the graph input: 0.8707 (hurts).
- Family combination (`exp/hyb/combo_cv.py`, our OOF decoded labels for 18 sessions mapped to their rows): probability
  product hurts; confidence gate (their calibrated confidence < 0.6 -> our label) +0.005 in CV -> LB **0.90954** (rank 5).
  Gate with the "our label must be their top-2" rule at 0.7 (CV +0.007) -> 0.90852 on the public LB.
- Second batch (4 subs): our labels inside their graph stage (`exp/hyb/graph_lab.py`): pseudo-label prior in the
  self-training rounds (CV +0.005 at 0.3), Sinkhorn count targets from our per-subject class counts (+0.003 at 0.5, hurts
  at 1.0), both on top of the top-2 gate: CV 0.8862 (gate 0.6) / 0.8871 (gate 0.55) vs plain gate 0.8839.
  LB: prior+counts+gate 0.6 = 0.91021; gate on the 3-run average P 0.90899; majority vote 0.90959; gate 0.55 = **0.91044**.
  Class-conditional gating (pair table, split-half checked) and per-subject quantile gating: +0.0005 in CV, not used.
- Board end of day: Nicolas 0.93334, Soheil 0.92154, localAI 0.91675, Mateo 0.91484, us 0.91044 (rank 5).

## 2026-09-29 (3 submissions)
- Error anatomy with true order (`exp/hyb/err_anatomy.py`): 95.4% of activity bouts have the correct majority label;
  bout-majority oracle 0.9718 vs 0.8827; errors 34% boundary, 47% scattered inside bouts, 19% whole-bout.
- Four subagents (`exp/hyb/agents/`): research (test session structure in participant_meta_data.txt: sbj_22/23 = 9+8+1
  activities, sbj_24/25 = 9+9; each 9-activity session is one protocol block; no public hints from the leaders);
  time-order seriation fails (median error 270-550 s, needs < 45 s); block/session prior +0.0018 at best (block already
  right for 98.6% of activity windows; video clusters follow posture, not session); bout clustering below baseline, kNN
  ICM relabelling +0.0053 alone but +0.0008 on top of the gate recipe.
- Link density in their graph stage (`exp/hyb/graph_lab.py`): union with our chain links CV 0.8871 -> 0.8882, LB
  **0.91289**; adding our top-2 candidate successors (CV 0.8898) -> 0.90957; adding their L0 links -> 0.90955.
  One-to-one assignment links transfer, candidate lists do not; changes of ~1% of labels move the public score by 0.003.
- Board: Mateo 0.93661, Nicolas 0.93334, localAI 0.92368, Anonym 0.92232, Soheil 0.92154, Free Chicken 0.92142, us 0.91289.

## 2026-10-03 (3 submissions)
- 2025-challenge test.csv (user joined the 2nd challenge; same subjects 22-25): it is ALL FOUR LIMBS of every one of our
  12,234 seconds as non-overlapping, unaugmented 1-s tiles with shuffled ids (`exp/hyb/w25_parse.py`, `w25_match.py`:
  85% of our tiles have an exact unique twin). Not an adjacency oracle. Ceiling study (`fuse_ceiling.py`, `fuse_lr.py`):
  with PERFECT cross-limb grouping, 4-limb sample continuity fused with the L2 link gives 95.8% exact successors and
  CV 0.9258 (graph) / 0.9327 (chain decoder); but 10% wrong groupings -> 0.67-0.78, and the grouping itself is
  impossible: the four sensors are not sample-synchronised (lag wanders +-75 samples within a session), so a pair
  classifier finds the right second 0.2-5% of the time (`exp/hyb/agents/group/`). No submission used the 2025 data.
- Decoder ceiling with true order (L2-faithful CV, `order_decode.py --links true --per rec`): 0.946; by exact-link
  share: 56% 0.872, 79% 0.900, 92% 0.918, 96% 0.927, 99% 0.933.
- Submissions: gl18 = gl11 recipe + the second fit's L2 assignment links as an extra one-to-one link set **0.91478**
  (new best); gl16 (geometric-mean base of two fits + second-fit links) 0.91316; gl19 (gl18 + our all-sessions
  link scorer as a third set) 0.91277. Board: Anonym 0.93864, Nicolas 0.93849, Mateo 0.93661, Santiago 0.92874.

## 2026-10-04
- Board at start: Anonym 0.94201, Nicolas 0.93849, Mateo 0.93661, Santiago 0.93593; us 0.92738 (rank 7). goodpjw2008
  published v3/v4 of "Learned Links + Counts" (LB 0.92942 from their own machine; L3 links with two two-tower seeds and
  three scorer seeds, kNN smoothing 1.5, boundary refiner, optional label cleaning / whitening).
- Host clarified (discussion 743256): prizes were fixed at the July 5 report deadline.
- Kaggle limits: 10 submissions/day, at most 2 concurrent batch GPU sessions.
- `kaggle/v4_fork/`: v4 forks that keep all decode inputs; heavier "big"/"opt" variants. A review workflow found that
  forcing ABLATION off made stage O4 raise KeyError (three ablation-only scores read unconditionally): fixed with
  `.get()` before any fork reached O4; the two affected runs were stopped and relaunched.
- `exp/v4/v4_local.py` reproduces the kernel decode exactly; `v4_combine.py` late fusion; `relink.py` link fusion.
- Submissions: public v4 Kaggle run 0.92444 (OOF 0.9272; same code scored 0.92942 on the author's machine: run-to-run
  noise); 10-03 fork with g1.5 + training-axis whitening + exercise identity 0.92513 (OOF 0.9287); same with
  per-subject whitening 0.92758 (OOF 0.9296); late fusion of both 10-03 fits, per-subject whitened, **0.92880**
  (OOF 0.9298); + public v4 run's Q 0.92847 (OOF 0.9304).
- OOF error anatomy of the best fusion: null<->activity at bout edges dominates; sibling swaps bench-dips<->lunges
  (complex), push-ups<->push-ups (complex); subjects 10 and 2 carry the unlabeled-tail sessions, subject 14 one whole-bout
  swap.

## 2026-10-04 evening / 2026-10-05
- Heavy "opt" forks (8 fusion + 4 IMU window seeds or 6 + 2 transformer-over-frames, four two-tower seeds, five scorer
  seeds, per-subject whitening + exercise-identity counts inside the notebook): OOF 0.9311 each after the refiner;
  their own submission.csv files scored **0.93208** (big-pool-opt-s7) and 0.92950 (big-tf-opt-s9). Board rank 5.
- Late fusion + refiner on CPU: s7+s9 OOF 0.9330; + two 10-03 fits OOF 0.9336, LB 0.93174; + public v4 run 0.9333;
  weights 0.3/0.3/0.2/0.2 0.9334.
- Research sweep (5 agents, nested OOF, baseline 0.9330): activity-transition prior in the refiner 0.9325-0.9328;
  refiner variants (per-fit, wider window, link/vote features) 0.9321-0.9332; null-edge specialist 0.9320-0.9322;
  fused link sets (relink.py, exact successor 0.6636 -> 0.6674) 0.9291-0.9294 per fit vs 0.9290-0.9301. None kept.
- Majority votes of the decodes: OOF 0.9340-0.9343, but LB 0.93077 (5 sources) and 0.92969 (3 sources).
  OOF differences below ~0.003 do not carry to the public LB; best public remains the single s7 run (0.93208).
- Short GPU jobs (`kaggle/v4_fork/patch_wseed.py`, ~16-22 min): extra fusion window seeds merged into s7's window blend
  on Kaggle (s7's per-run outputs mounted as a kernel source), then `exp/v4/apply_wseed.py` rebuilds stage-B
  (0.2 window + 0.5 S3 + 0.3 adapted T) and the 4-run fusion + refiner is re-run on CPU.
  Job a (2 pool seeds): window F1 0.7264 -> 0.7267, fusion OOF 0.9336, **LB 0.93278** (best).
  Job b (+2 transformer-over-frames seeds): window F1 -> 0.7311, fusion OOF 0.9331, LB 0.93187.
  The window models carry only 20% of the stage-B base, so window-level gains barely reach the final labels.

## 2026-10-02 (3 submissions)
- Per-subject cross-fitted self-training (`exp/hyb/selftrain_cv.py`): a small LightGBM per subject trained on the
  pipeline's own pseudo-labels (5-fold over the subject's windows), multiplied into the graph output. Alone 0.859, as a
  correction at weight 0.08: CV 0.8882 -> 0.8919, LB **0.91367**; weight 0.15: CV 0.8914, LB 0.91197.
- Transductive tabular expert (`exp/hyb/adapt_tab.py`): the Hanbat T expert retrained on train + cross-fitted target
  pseudo-labels; alone 0.829 OOF (plain T 0.640); as graph input 0.5 + post-graph 0.15: CV 0.8955, LB 0.90852.
- Pattern across all days: additions of our independent evidence (our links, our labels, light self-training) transfer
  to the public LB; changes that reinforce the Hanbat pipeline's own beliefs gain in CV and lose on the public subset.
- Board: Nicolas 0.93849, Mateo 0.93661, Anonym 0.93200, Santiago 0.92874, Free Chicken 0.92554, localAI 0.92368,
  Soheil 0.92154, us 0.91367 (rank 8).
- Second batch (2 subs). LB-faithful CV: fold-honest OOF L2 links (`work/hanbat/l2oof/oof_L2.npz`, exact successor 0.56,
  same-label 0.94; `graph_lab.py cv --links L2`) reproduce the base at 0.9045 (LB 0.9058). Under it every recipe after
  prior+gate is flat within noise (0.9042-0.9072): the pipeline family is on a plateau. New public notebook
  honghanhhh "LB 0.9" is the same Hanbat pipeline (0.90770). Second independent fit of the base from the public fitted
  models (`kaggle/hanbat_run2`, 3 min GPU): our recipe on it 0.91344; probability average of both fits 0.91272.

## Earlier log (cloud sandbox session, 2026-09-22)

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
